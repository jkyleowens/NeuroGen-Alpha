#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Enhanced error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Check for errors after kernel launches
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA KERNEL ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// Memory allocation with error checking
template<typename T>
inline void safeCudaMalloc(T** ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
}

template<typename T>
inline void safeCudaMemcpy(T* dst, const T* src, size_t count, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), kind));
}

template<typename T>
inline void safeCudaMemset(T* ptr, int value, size_t count) {
    CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
}

// Device info utilities
inline void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
}

// Optimal grid/block configuration
inline dim3 getOptimalBlockSize() {
    return dim3(256); // Good balance for most kernels
}

inline dim3 getOptimalGridSize(int total_elements, int block_size = 256) {
    return dim3((total_elements + block_size - 1) / block_size);
}

#endif // CUDA_UTILS_CUH