#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr NSUInteger kPageBytes = 64 * 1024;

std::string jsonEscape(const char *value) {
  std::string result;
  if (!value) return result;
  for (const unsigned char ch : std::string(value)) {
    switch (ch) {
    case '\\': result += "\\\\"; break;
    case '"': result += "\\\""; break;
    case '\n': result += "\\n"; break;
    case '\r': result += "\\r"; break;
    case '\t': result += "\\t"; break;
    default:
      if (ch < 0x20) {
        std::ostringstream escaped;
        escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned int>(ch);
        result += escaped.str();
      } else {
        result += static_cast<char>(ch);
      }
    }
  }
  return result;
}

std::string jsonString(NSString *value) {
  return "\"" + jsonEscape(value ? value.UTF8String : "") + "\"";
}

std::string errorString(NSError *error) {
  return error ? (error.localizedDescription.UTF8String ?: "unknown error")
               : "unknown error";
}

uint32_t gpuCoreCount(uint64_t registryID) {
  uint32_t result = 0;
  auto read = [&](io_registry_entry_t entry) {
    if (!entry) return;
    CFTypeRef value = IORegistryEntryCreateCFProperty(
        entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (value) {
      int64_t count = 0;
      if (CFGetTypeID(value) == CFNumberGetTypeID() &&
          CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberSInt64Type,
                           &count) &&
          count > 0 && count <= 4096) {
        result = static_cast<uint32_t>(count);
      }
      CFRelease(value);
    }
  };

  io_registry_entry_t entry = IOServiceGetMatchingService(
      kIOMainPortDefault, IORegistryEntryIDMatching(registryID));
  for (int depth = 0; entry && depth < 4 && !result; ++depth) {
    read(entry);
    if (result) break;
    io_registry_entry_t parent = MACH_PORT_NULL;
    if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
        KERN_SUCCESS) {
      parent = MACH_PORT_NULL;
    }
    IOObjectRelease(entry);
    entry = parent;
  }
  if (entry) IOObjectRelease(entry);

  if (!result) {
    entry = IOServiceGetMatchingService(kIOMainPortDefault,
                                        IOServiceMatching("IOAccelerator"));
    if (entry) {
      read(entry);
      IOObjectRelease(entry);
    }
  }
  return result;
}

bool placementSparseSupport(id<MTLDevice> device) {
  if (@available(macOS 26.4, *)) {
    const SEL selector = sel_registerName("supportsPlacementSparse");
    if (![device respondsToSelector:selector]) return false;
    using Query = BOOL (*)(id, SEL);
    return reinterpret_cast<Query>(objc_msgSend)(device, selector);
  }
  return false;
}

struct PipelineResult {
  bool compiled = false;
  bool executed = false;
  NSUInteger threadExecutionWidth = 0;
  NSUInteger maxThreads = 0;
  NSUInteger staticThreadgroupBytes = 0;
  std::string error;
};

PipelineResult makePipeline(id<MTLDevice> device, id<MTLLibrary> library,
                            NSString *name, NSUInteger dispatchThreads,
                            id<MTLCommandQueue> queue) {
  PipelineResult result;
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) {
    result.error = "function not found";
    return result;
  }
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    result.error = errorString(error);
    return result;
  }
  result.compiled = true;
  result.threadExecutionWidth = pipeline.threadExecutionWidth;
  result.maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
  result.staticThreadgroupBytes = pipeline.staticThreadgroupMemoryLength;

  id<MTLBuffer> output = [device newBufferWithLength:dispatchThreads * 4
                                              options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!output || !command || !encoder) {
    result.error = "pipeline execution resources unavailable";
    return result;
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  const NSUInteger width = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
  const NSUInteger threads = std::min(dispatchThreads,
                                      pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(dispatchThreads, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(std::max(width, threads), 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    result.error = command.error ? errorString(command.error)
                                 : "command buffer did not complete";
    return result;
  }
  result.executed = true;
  return result;
}

struct SparseResult {
  std::string status = "skipped";
  bool mapped = false;
  bool copied = false;
  bool unmapped = false;
  std::string error;
};

SparseResult sparseProbe(id<MTLDevice> device) {
  SparseResult result;
  if (!placementSparseSupport(device)) return result;
  if (@available(macOS 26.0, *)) {
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTLSharedEvent> event = [device newSharedEvent];
    id<MTLCommandQueue> copyQueue = [device newCommandQueue];
    id<MTLBuffer> source = [device newBufferWithLength:kPageBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> destination =
        [device newBufferWithLength:kPageBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> sparse = [device
        newBufferWithLength:kPageBytes
                    options:MTLResourceStorageModePrivate
     placementSparsePageSize:MTLSparsePageSize64];
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = kPageBytes;
    descriptor.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize64;
    id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
    if (!queue || !event || !copyQueue || !source || !destination || !sparse ||
        !heap) {
      result.status = "failed";
      result.error = "sparse probe resource creation failed";
      return result;
    }
    auto *sourceBytes = static_cast<uint32_t *>(source.contents);
    auto *destinationBytes = static_cast<uint32_t *>(destination.contents);
    for (NSUInteger i = 0; i < kPageBytes / sizeof(uint32_t); ++i) {
      sourceBytes[i] = 0xA5A50000u + static_cast<uint32_t>(i);
      destinationBytes[i] = 0;
    }
    MTL4UpdateSparseBufferMappingOperation operation{};
    operation.mode = MTLSparseTextureMappingModeMap;
    operation.bufferRange = NSMakeRange(0, 1);
    operation.heapOffset = 0;
    [queue updateBufferMappings:sparse heap:heap operations:&operation count:1];
    [queue signalEvent:event value:1];
    result.mapped = [event waitUntilSignaledValue:1 timeoutMS:5000];
    if (result.mapped) {
      id<MTLCommandBuffer> command = [copyQueue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
      [blit copyFromBuffer:source
              sourceOffset:0
                  toBuffer:sparse
           destinationOffset:0
                        size:kPageBytes];
      [blit copyFromBuffer:sparse
              sourceOffset:0
                  toBuffer:destination
           destinationOffset:0
                        size:kPageBytes];
      [blit endEncoding];
      [command commit];
      [command waitUntilCompleted];
      result.copied = command.status == MTLCommandBufferStatusCompleted &&
                      std::memcmp(source.contents, destination.contents,
                                  kPageBytes) == 0;
    }
    operation.mode = MTLSparseTextureMappingModeUnmap;
    [queue updateBufferMappings:sparse heap:nil operations:&operation count:1];
    [queue signalEvent:event value:2];
    result.unmapped = [event waitUntilSignaledValue:2 timeoutMS:5000];
    result.status = result.mapped && result.copied && result.unmapped
                        ? "pass"
                        : "failed";
    if (result.status == "failed") result.error = "sparse map/copy/unmap check failed";
  }
  return result;
}

void printPipeline(std::ostringstream &out, const PipelineResult &result) {
  out << "{\"compiled\":" << (result.compiled ? "true" : "false")
      << ",\"executed\":" << (result.executed ? "true" : "false")
      << ",\"thread_execution_width\":" << result.threadExecutionWidth
      << ",\"max_threads_per_threadgroup\":" << result.maxThreads
      << ",\"static_threadgroup_memory_bytes\":"
      << result.staticThreadgroupBytes << ",\"error\":\""
      << jsonEscape(result.error.c_str()) << "\"}";
}

} // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cout << "{\"status\":\"failed\",\"error\":\"no Metal device\"}\n";
      return 1;
    }
    NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
    const uint32_t cores = gpuCoreCount(device.registryID);
    std::ostringstream out;
    out << "{\"status\":\"ok\",\"device_name\":" << jsonString(device.name)
        << ",\"registry_id\":" << device.registryID
        << ",\"unified_memory\":" << (device.hasUnifiedMemory ? "true" : "false")
        << ",\"gpu_core_count\":" << cores
        << ",\"physical_memory_bytes\":"
        << NSProcessInfo.processInfo.physicalMemory
        << ",\"recommended_max_working_set_bytes\":"
        << device.recommendedMaxWorkingSetSize
        << ",\"max_buffer_length_bytes\":" << device.maxBufferLength
        << ",\"max_threadgroup_memory_bytes\":"
        << device.maxThreadgroupMemoryLength
        << ",\"max_threads_per_threadgroup\":{\"width\":"
        << device.maxThreadsPerThreadgroup.width << ",\"height\":"
        << device.maxThreadsPerThreadgroup.height << ",\"depth\":"
        << device.maxThreadsPerThreadgroup.depth << "}"
        << ",\"macos_version\":" << version.majorVersion << '.'
        << version.minorVersion << '.' << version.patchVersion;

    uint32_t highestFamily = 0;
    out << ",\"apple_families\":{";
    for (uint32_t family = 7; family <= 10; ++family) {
      const bool supported = [device supportsFamily:
          static_cast<MTLGPUFamily>(1000 + family)];
      if (supported) highestFamily = std::max(highestFamily, family);
      if (family != 7) out << ',';
      out << "\"apple" << family << "\":"
          << (supported ? "true" : "false");
    }
    out << "},\"highest_apple_family\":" << highestFamily
        << ",\"supports_placement_sparse\":"
        << (placementSparseSupport(device) ? "true" : "false");

    out << ",\"counter_sets\":[";
    if (device.counterSets) {
      for (NSUInteger i = 0; i < device.counterSets.count; ++i) {
        if (i) out << ',';
        out << jsonString(device.counterSets[i].name);
      }
    }
    out << "],\"counter_sampling\":{";
    const std::vector<std::pair<const char *, MTLCounterSamplingPoint>> points{
        {"stage", MTLCounterSamplingPointAtStageBoundary},
        {"draw", MTLCounterSamplingPointAtDrawBoundary},
        {"dispatch", MTLCounterSamplingPointAtDispatchBoundary},
        {"tile_dispatch", MTLCounterSamplingPointAtTileDispatchBoundary},
        {"blit", MTLCounterSamplingPointAtBlitBoundary}};
    for (size_t i = 0; i < points.size(); ++i) {
      if (i) out << ',';
      out << '"' << points[i].first << "\":"
          << ([device supportsCounterSampling:points[i].second] ? "true" : "false");
    }
    out << "}";

    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    NSString *source = @"#include <metal_stdlib>\n"
                        "using namespace metal;\n"
                        "kernel void probe_bfloat(device bfloat *out [[buffer(0)]], "
                        "uint id [[thread_position_in_grid]]) { out[id] = bfloat(id); }\n"
                        "kernel void probe_simd_matrix(device float *out [[buffer(0)]], "
                        "uint id [[thread_index_in_threadgroup]]) { "
                        "simdgroup_matrix<float, 8, 8> m; if (id == 0) out[0] = 1.0f; }\n";
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
    out << ",\"runtime_shader_library\":{";
    if (!library) {
      out << "\"compiled\":false,\"error\":\""
          << jsonEscape(errorString(error).c_str()) << "\"}";
    } else {
      id<MTLCommandQueue> queue = [device newCommandQueue];
      out << "\"compiled\":true,\"error\":\"\",\"bfloat_pipeline\":";
      printPipeline(out, makePipeline(device, library, @"probe_bfloat", 64, queue));
      out << ",\"simd_matrix_pipeline\":";
      printPipeline(out, makePipeline(device, library, @"probe_simd_matrix", 32, queue));
      out << '}';
    }

    const SparseResult sparse = sparseProbe(device);
    out << ",\"sparse_probe\":{\"status\":\""
        << jsonEscape(sparse.status.c_str()) << "\",\"mapped\":"
        << (sparse.mapped ? "true" : "false") << ",\"copied\":"
        << (sparse.copied ? "true" : "false") << ",\"unmapped\":"
        << (sparse.unmapped ? "true" : "false") << ",\"error\":\""
        << jsonEscape(sparse.error.c_str()) << "\"}";
    out << "}\n";
    std::cout << out.str();
    return 0;
  }
}
