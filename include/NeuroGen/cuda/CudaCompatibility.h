#ifndef CUDA_COMPATIBILITY_H
#define CUDA_COMPATIBILITY_H

// Always include real CUDA headers when building with nvcc
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cstdio>    // For fprintf, stderr
#include <cstdlib>   // For exit
#include <iostream>  // For std::cerr

// CUDA version compatibility checks - Made more lenient
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
#warning "CUDA version 9.0 or higher is recommended for best compatibility"
#endif

namespace cuda_compat {
    // Provided examples:
    template<typename T> struct is_array { static constexpr bool value = false; }; // Note: The document shows 'false' as a placeholder.
                                                                                 // A more complete implementation might use __is_array(T) if available
                                                                                 // or more sophisticated checks.
    template<typename T> struct is_reference { static constexpr bool value = false; }; // Similar placeholder.

    // ... additional type traits
    // Based on the "Root Cause Analysis" which mentions `__is_member_object_pointer`,
    // a placeholder for it would look like:
    // template<typename T> struct is_member_object_pointer { static constexpr bool value = false; }; // Placeholder

    // Other type traits that caused issues like those related to `__type_pack_element`
    // would follow a similar pattern, providing a definition that NVCC can compile.
    // For instance, if a specific trait like `std::is_void<T>::value` was problematic,
    // it might be re-implemented or specialized here.
    // The exact list of "... additional type traits" would depend on all specific
    // incompatibilities encountered during compilation.
}

// Device function attributes for different CUDA versions
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ < 600
        #define CUDA_ARCH_COMPATIBLE 0
    #else
        #define CUDA_ARCH_COMPATIBLE 1
    #endif
#else
    #define CUDA_ARCH_COMPATIBLE 1
#endif

// Thread synchronization compatibility
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 700
        #define CUDA_SYNC() __syncwarp()
    #else
        #define CUDA_SYNC() __syncthreads()
    #endif
#else
    #define CUDA_SYNC() __syncthreads()
#endif

// Atomic operations compatibility
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 600
        #define CUDA_ATOMIC_ADD_FLOAT(addr, val) atomicAdd(addr, val)
    #else
        __device__ inline float atomicAddFloat(float* address, float val) {
            int* address_as_i = (int*)address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
            } while (assumed != old);
            return __int_as_float(old);
        }
        #define CUDA_ATOMIC_ADD_FLOAT(addr, val) atomicAddFloat(addr, val)
    #endif
#else
    #define CUDA_ATOMIC_ADD_FLOAT(addr, val) atomicAdd(addr, val)
#endif

// Memory alignment macros
#define CUDA_ALIGN(x) __align__(x)
#define CUDA_SHARED __shared__
#define CUDA_CONSTANT __constant__

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// Grid and block dimension helpers
#define CUDA_THREADS_PER_BLOCK 256
#define CUDA_GET_BLOCKS(n) (((n) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)

// Note: makeSafeBlock and makeSafeGrid functions are now defined in CudaUtils.h


// Host-device function decorators
#ifdef __CUDACC__
    #define CUDA_DEVICE __device__
    #define CUDA_HOST __host__
    #define CUDA_GLOBAL __global__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_DEVICE
    #define CUDA_HOST
    #define CUDA_GLOBAL
    #define CUDA_HOST_DEVICE
#endif

// Inline function attribute
#ifdef __CUDACC__
    #define CUDA_INLINE __forceinline__
#else
    #define CUDA_INLINE inline
#endif

// Mathematical constants compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Random number generation compatibility
typedef curandState_t CudaRandState;

// Memory management helpers
template<typename T>
CUDA_HOST_DEVICE inline T* cuda_malloc(size_t count) {
    T* ptr;
    #ifdef __CUDA_ARCH__
        // Device code - cannot allocate memory
        return nullptr;
    #else
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        return ptr;
    #endif
}

template<typename T>
CUDA_HOST_DEVICE inline void cuda_free(T* ptr) {
    #ifdef __CUDA_ARCH__
        // Device code - cannot free memory
    #else
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    #endif
}

// Texture memory compatibility (if needed)
#if CUDA_VERSION >= 11000
    #define CUDA_TEXTURE_SUPPORT 1
#else
    #define CUDA_TEXTURE_SUPPORT 0
#endif

// Cooperative groups compatibility
#if CUDA_VERSION >= 9000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    #include <cooperative_groups.h>
    #define CUDA_COOPERATIVE_GROUPS_SUPPORT 1
    namespace cg = cooperative_groups;
#else
    #define CUDA_COOPERATIVE_GROUPS_SUPPORT 0
#endif

// Half precision support
#if CUDA_VERSION >= 7050 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    #include <cuda_fp16.h>
    #define CUDA_HALF_SUPPORT 1
    typedef __half CudaHalf;
#else
    #define CUDA_HALF_SUPPORT 0
    typedef float CudaHalf;  // Fallback to float
#endif

// Utility functions for device properties
inline int getCudaDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline cudaDeviceProp getCudaDeviceProperties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

// Stream management
inline cudaStream_t createCudaStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

inline void destroyCudaStream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Event management
inline cudaEvent_t createCudaEvent() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

inline void destroyCudaEvent(cudaEvent_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

#endif // CUDA_COMPATIBILITY_H