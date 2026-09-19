#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kRows = 8;
constexpr uint32_t kMaximumBatch = 4;
constexpr uint32_t kInput = 5120;
constexpr uint32_t kOutput = 16640;
constexpr uint32_t kGroups = 60;
constexpr uint32_t kQuantGroup = 64;

[[noreturn]] void fail(const std::string &message);

uint32_t unpackNibble(const uint8_t *weights, uint32_t index) {
  const uint8_t packed = weights[index / 2];
  return (index & 1) ? packed >> 4 : packed & 0x0f;
}

uint32_t bfloatUlpDistance(uint16_t actual, uint16_t expected) {
  if ((actual ^ expected) & 0x8000)
    return UINT32_MAX;
  return actual > expected ? actual - expected : expected - actual;
}

void checkNibbleBoundaries() {
  std::array<uint8_t, 32> packed{};
  for (uint32_t index = 0; index < 64; ++index)
    packed[index / 2] = static_cast<uint8_t>(
        (index & 1) ? (packed[index / 2] & 0x0f) | ((index % 16) << 4)
                    : (packed[index / 2] & 0xf0) | (index % 16));
  for (uint32_t index = 0; index < 64; ++index)
    if (unpackNibble(packed.data(), index) != index % 16)
      fail("Q4 nibble boundary unpack differs from handcrafted layout");
}

void scalarQ4Reference(const __bf16 *input, const uint8_t *weights,
                       const __bf16 *scales, const __bf16 *biases,
                       __bf16 *output, uint32_t rows, uint32_t outputSize,
                       uint32_t inputSize) {
  const uint32_t quantGroups = inputSize / kQuantGroup;
  const uint32_t storageN = 256;
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < outputSize; ++column) {
      const uint32_t tile = column / storageN;
      const uint32_t tileOffset = column % storageN;
      float value = 0.0f;
      for (uint32_t group = 0; group < quantGroups; ++group) {
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint32_t index = 0; index < kQuantGroup; ++index) {
          const float inputValue = static_cast<float>(
              input[row * inputSize + group * kQuantGroup + index]);
          const uint32_t weightColumn =
              tile * quantGroups * storageN + group * storageN + tileOffset;
          dot = std::fmaf(
              inputValue,
              static_cast<float>(unpackNibble(weights + weightColumn * 32,
                                              index)),
              dot);
          sum += inputValue;
        }
        const uint32_t parameter =
            (tile * quantGroups + group) * storageN + tileOffset;
        value = std::fmaf(dot, static_cast<float>(scales[parameter]),
                          sum * static_cast<float>(biases[parameter])) + value;
      }
      output[row * outputSize + column] = __bf16(value);
    }
  }
}

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

ComputeDispatch affine(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer output,
                       const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(output)}};
  result.bytes = {{5, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch gateUp(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer output,
                       const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, weights},
                    {2, scales},
                    {3, biases},
                    {4, std::move(output)},
                    {5, std::move(weights)},
                    {6, std::move(scales)},
                    {7, std::move(biases)}};
  result.bytes = {{8, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch upSilu(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer gate,
                       MetalBuffer output, const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(gate)},
                    {5, std::move(output)}};
  result.bytes = {{6, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

void run(const std::string &metallibPath) {
  checkNibbleBoundaries();
  MetalBackend backend(metallibPath);
  const uint64_t inputElements = uint64_t{kRows} * kInput;
  const uint64_t outputElements = uint64_t{kRows} * kOutput;
  const uint64_t weightElements = uint64_t{kInput} * kOutput;
  const uint64_t parameterElements = weightElements / kQuantGroup;

  MetalBuffer input = shared(
      backend, kMaximumBatch * inputElements * sizeof(__bf16), "q4-input");
  MetalBuffer weights = shared(backend, weightElements / 2, "q4-weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), "q4-scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), "q4-biases");
  MetalBuffer reference =
      shared(backend, kMaximumBatch * outputElements * sizeof(__bf16),
             "q4-reference");

  std::mt19937 random(7319);
  std::uniform_real_distribution<float> inputValues(-1.0f, 1.0f);
  std::uniform_real_distribution<float> parameters(-0.02f, 0.02f);
  auto *inputValuesPtr = static_cast<__bf16 *>(input.contents());
  for (uint64_t index = 0; index < kMaximumBatch * inputElements; ++index)
    inputValuesPtr[index] = __bf16(inputValues(random));
  auto *weight = static_cast<uint8_t *>(weights.contents());
  for (uint64_t index = 0; index < weightElements / 2; ++index)
    weight[index] = static_cast<uint8_t>(random());
  auto *scale = static_cast<__bf16 *>(scales.contents());
  auto *bias = static_cast<__bf16 *>(biases.contents());
  for (uint64_t index = 0; index < parameterElements; ++index) {
    scale[index] = __bf16(parameters(random));
    bias[index] = __bf16(parameters(random));
  }
  std::memset(reference.contents(), 0, reference.sizeBytes());

  // Every Q4 projection has one StorageN=256 representation. These compute
  // kernels consume it with TileN=128 for the four fixed DFlash batch widths.
  const Q4Params params{kOutput, kInput, kGroups};
  std::vector<ComputeDispatch> singles;
  std::memset(reference.contents(), 0, reference.sizeBytes());
  singles.clear();
  for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
    singles.push_back(affine(
        "decode_linear_q4_n128",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(reference,
                     uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params));
  }
  (void)backend.submitCommand(singles);

  std::vector<__bf16> scalarReference(kMaximumBatch * outputElements);
  scalarQ4Reference(inputValuesPtr, weight, scale, bias, scalarReference.data(),
                    kMaximumBatch * kRows, kOutput, kInput);
  const auto *actual = static_cast<const __bf16 *>(reference.contents());
  uint64_t mismatches = 0;
  uint32_t maxUlp = 0;
  float maxAbsoluteError = 0.0f;
  double sumAbsoluteError = 0.0;
  size_t firstMismatch = scalarReference.size();
  uint16_t firstActualBits = 0;
  uint16_t firstExpectedBits = 0;
  for (size_t index = 0; index < scalarReference.size(); ++index) {
    if (actual[index] == scalarReference[index])
      continue;
    ++mismatches;
    uint16_t actualBits = 0;
    uint16_t expectedBits = 0;
    std::memcpy(&actualBits, actual + index, sizeof(actualBits));
    std::memcpy(&expectedBits, scalarReference.data() + index,
                sizeof(expectedBits));
    maxUlp = std::max(maxUlp, bfloatUlpDistance(actualBits, expectedBits));
    const float actualValue = static_cast<float>(actual[index]);
    const float expectedValue = static_cast<float>(scalarReference[index]);
    if (!std::isfinite(actualValue) || !std::isfinite(expectedValue))
      fail("Apple7 Q4 projection produced a non-finite scalar-reference value");
    const float absoluteError = std::fabs(actualValue - expectedValue);
    maxAbsoluteError = std::max(maxAbsoluteError, absoluteError);
    sumAbsoluteError += absoluteError;
    if (firstMismatch == scalarReference.size()) {
      firstMismatch = index;
      firstActualBits = actualBits;
      firstExpectedBits = expectedBits;
    }
  }
  // The Metal compiler may fuse FP32 arithmetic differently from the host
  // scalar reference before the final BF16 conversion. Keep the independent
  // oracle strict in value space while allowing that documented rounding
  // difference; layout or nibble errors are orders of magnitude larger.
  const double meanAbsoluteError =
      sumAbsoluteError / static_cast<double>(scalarReference.size());
  if (maxAbsoluteError > 0.125f || !std::isfinite(maxAbsoluteError)) {
    std::cerr << "Q4 first mismatch index=" << firstMismatch
              << " actual_bits=0x" << std::hex << firstActualBits
              << " expected_bits=0x" << firstExpectedBits << std::dec
              << " actual=" << static_cast<float>(actual[firstMismatch])
              << " expected="
              << static_cast<float>(scalarReference[firstMismatch])
              << " mismatches=" << mismatches << " max_ulp=" << maxUlp
              << " max_abs=" << maxAbsoluteError
              << " mean_abs=" << meanAbsoluteError << '\n';
    fail("Apple7 Q4 projection differs from the independent scalar reference");
  }
  if (mismatches) {
    std::cout << "PASS q4 scalar reference max_abs=" << maxAbsoluteError
              << " mean_abs=" << meanAbsoluteError
              << " mismatches=" << mismatches << " max_ulp=" << maxUlp
              << '\n';
  } else {
    std::cout << "PASS q4 scalar reference exact=true\n";
  }

  // The pipelined narrow-projection kernel issues two quant groups before
  // either epilogue; its outputs must be byte-identical to the sequential M8.
  MetalBuffer paired = shared(backend, kMaximumBatch * outputElements *
                                           sizeof(__bf16),
                              "q4-paired-output");
  std::memset(paired.contents(), 0, paired.sizeBytes());
  std::vector<ComputeDispatch> pairedSingles;
  for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
    pairedSingles.push_back(affine(
        "decode_linear_q4_n128_paired",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(paired, uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params));
  }
  (void)backend.submitCommand(pairedSingles);
  if (std::memcmp(reference.contents(), paired.contents(), paired.sizeBytes()))
    fail("paired M8 projection differs from the sequential M8 projection");
  std::cout << "PASS q4 paired M8 exact=true\n";
  constexpr std::array<const char *, 3> genericPipelines{
      "decode_linear_q4_n128_m16", "decode_linear_q4_n128_m24",
      "decode_linear_q4_n128_m32"};
  for (uint32_t width = 2; width <= kMaximumBatch; ++width) {
    MetalBuffer candidate =
        shared(backend, uint64_t{width} * outputElements * sizeof(__bf16),
               "q4-generic-batch-output");
    std::memset(candidate.contents(), 0, candidate.sizeBytes());
    ComputeDispatch batch = affine(
        genericPipelines[width - 2],
        backend.view(input, 0,
                     uint64_t{width} * inputElements * sizeof(__bf16)),
        weights, scales, biases, candidate, params);
    const auto timing = backend.submitCommand({&batch, 1});
    const uint64_t comparedBytes =
        uint64_t{width} * outputElements * sizeof(__bf16);
    if (std::memcmp(reference.contents(), candidate.contents(), comparedBytes))
      fail("generic M" + std::to_string(width * kRows) +
           " projection differs from its M8 references");
    std::cout << "PASS q4 generic M" << width * kRows
              << " exact=true wall_seconds=" << timing.wallSeconds << '\n';
  }

  const uint64_t m24Bytes = uint64_t{3} * outputElements * sizeof(__bf16);
  MetalBuffer gateUpReference =
      shared(backend, m24Bytes, "q4-m8-gate-up-reference");
  MetalBuffer gateScratch = shared(backend, m24Bytes, "q4-m24-gate");
  MetalBuffer combined = shared(backend, m24Bytes, "q4-m24-up-silu");
  std::array<ComputeDispatch, 3> gateUpSingles;
  for (uint32_t lane = 0; lane < gateUpSingles.size(); ++lane) {
    gateUpSingles[lane] = gateUp(
        "decode_linear_q4_n256_gate_up",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(gateUpReference,
                     uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params);
  }
  (void)backend.submitCommand(gateUpSingles);
  std::array<ComputeDispatch, 2> splitDispatches{
      affine("decode_linear_q4_n256_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, params),
      upSilu("decode_linear_q4_n256_up_silu_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, combined, params)};
  const auto timing = backend.submitCommand(splitDispatches);
  if (std::memcmp(gateUpReference.contents(), combined.contents(), m24Bytes))
    fail("M24 split gate/up differs from its M8 references");
  std::cout << "PASS q4 M24 split-gate exact=true wall_seconds="
            << timing.wallSeconds << '\n';

  // A persistent threadgroup runs its tiles back-to-back on one input-sum
  // scratch: the next tile's prologue rewrites region 0, which the last
  // quant-group block still reads when K % 512 == 256. One threadgroup
  // striding over every tile must match one tile per threadgroup.
  constexpr uint32_t kPersistentOutput = 768;
  constexpr uint32_t kPersistentLanes = 3;
  for (const uint32_t persistentInput : {768u, 1280u}) {
    const uint64_t laneInputBytes =
        uint64_t{kRows} * persistentInput * sizeof(__bf16);
    const uint64_t laneOutputBytes =
        uint64_t{kRows} * kPersistentOutput * sizeof(__bf16);
    const uint64_t outputBytes = kPersistentLanes * laneOutputBytes;
    MetalBuffer singleTile = shared(backend, outputBytes, "q4-single-tile");
    MetalBuffer persistent = shared(backend, outputBytes, "q4-persistent");
    std::memset(singleTile.contents(), 0, outputBytes);
    std::memset(persistent.contents(), 0, outputBytes);
    const Q4Params oneTileEach{kPersistentOutput, persistentInput,
                               kPersistentOutput / 128};
    std::vector<ComputeDispatch> dispatches;
    for (uint32_t lane = 0; lane < kPersistentLanes; ++lane) {
      dispatches.push_back(affine(
          "decode_linear_q4_n128",
          backend.view(input, lane * laneInputBytes, laneInputBytes), weights,
          scales, biases,
          backend.view(singleTile, lane * laneOutputBytes, laneOutputBytes),
          oneTileEach));
    }
    const Q4Params oneGroup{kPersistentOutput, persistentInput, 1};
    dispatches.push_back(affine(
        "decode_linear_q4_n128_m24",
        backend.view(input, 0, kPersistentLanes * laneInputBytes), weights,
        scales, biases, persistent, oneGroup));
    (void)backend.submitCommand(dispatches);
    if (std::memcmp(singleTile.contents(), persistent.contents(), outputBytes))
      fail("persistent M24 projection at K=" + std::to_string(persistentInput) +
           " differs from its single-tile M8 references");
    std::cout << "PASS q4 persistent M24 K=" << persistentInput
              << " exact=true\n";
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_batched_projection_metal_test <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
