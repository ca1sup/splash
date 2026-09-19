#pragma once

#include "MetalBackend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

namespace splash::metal {

inline bool queryPlacementSparseSupport(id<MTLDevice> device) {
    if (@available(macOS 26.4, *)) {
        @try {
            // Xcode 26.2's SDK predates the property declaration, while the
            // macOS 27 driver can expose it.  Resolve the selector at runtime
            // so the Apple7 port can be built with either SDK generation.
            const SEL selector = sel_registerName("supportsPlacementSparse");
            if (![device respondsToSelector:selector]) return false;
            using Query = BOOL (*)(id, SEL);
            return reinterpret_cast<Query>(objc_msgSend)(device, selector);
        } @catch (NSException *exception) {
            // A driver wrapper may expose the selector but fail when forwarding it.
            throw MetalBackendError(
                std::string("Metal supportsPlacementSparse query failed: ") +
                (exception.name.UTF8String ?: "NSException") + ": " +
                (exception.reason.UTF8String ?: "unknown driver error"));
        }
    }
    return false;
}

} // namespace splash::metal
