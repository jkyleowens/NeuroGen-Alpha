#include <iostream>

// Mock CUDA types for testing without nvcc
#ifndef __CUDACC__
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

typedef int cudaError_t;
typedef void* cudaDeviceProp;
typedef void* curandState;

#define cudaSuccess 0
#define __host__
#define __device__
#define __global__

cudaError_t cudaMalloc(void** ptr, size_t size) { return cudaSuccess; }
cudaError_t cudaFree(void* ptr) { return cudaSuccess; }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
cudaError_t cudaGetDeviceCount(int* count) { *count = 1; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(void* prop, int device) { return cudaSuccess; }
cudaError_t cudaSetDevice(int device) { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t error) { return "Mock error"; }

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#endif

// Test include order to check for redefinition errors
#include "include/NeuroGen/cuda/CudaCompatibility.h"
#include "include/NeuroGen/cuda/GridBlockUtils.cuh"
#include "include/NeuroGen/cuda/CudaUtils.h"

int main() {
    std::cout << "Testing fixed headers for redefinition errors..." << std::endl;
    
    // Test functions from GridBlockUtils.cuh
    dim3 basic_block = makeBlock();  // From GridBlockUtils.cuh
    dim3 basic_grid = makeGrid(1000); // From GridBlockUtils.cuh
    
    // Test functions from CudaUtils.h namespace  
    dim3 safe_block = CudaUtils::makeSafeBlock(512);
    dim3 safe_grid = CudaUtils::makeSafeGrid(2000, 512);
    
    // Test global safe functions
    dim3 global_safe_block = makeSafeBlock(256);
    dim3 global_safe_grid = makeSafeGrid(1000, 256);
    
    std::cout << "Basic block dimensions: " << basic_block.x << std::endl;
    std::cout << "Basic grid dimensions: " << basic_grid.x << std::endl;
    std::cout << "Safe block dimensions: " << safe_block.x << std::endl;
    std::cout << "Safe grid dimensions: " << safe_grid.x << std::endl;
    std::cout << "Global safe block dimensions: " << global_safe_block.x << std::endl;
    std::cout << "Global safe grid dimensions: " << global_safe_grid.x << std::endl;
    
    std::cout << "✅ No redefinition errors! Headers fixed successfully." << std::endl;
    return 0;
}
