#include <iostream>

// Mock CUDA types for syntax testing when nvcc is not available
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

// Mock CUDA functions
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

// Test include of our headers
#include "src/cuda/CudaUtils.h"

int main() {
    std::cout << "Testing CudaUtils.h syntax..." << std::endl;
    
    // Test the utility functions
    dim3 block = makeBlock();
    dim3 grid = makeGrid(1000);
    dim3 safe_block = makeSafeBlock(256);
    dim3 safe_grid = makeSafeGrid(1000, 256);
    
    std::cout << "Block dimensions: " << block.x << std::endl;
    std::cout << "Grid dimensions: " << grid.x << std::endl;
    std::cout << "Safe block dimensions: " << safe_block.x << std::endl;
    std::cout << "Safe grid dimensions: " << safe_grid.x << std::endl;
    
    // Test namespace functions
    dim3 ns_block = CudaUtils::makeBlock(512);
    dim3 ns_grid = CudaUtils::makeGrid(2000, 512);
    
    std::cout << "Namespace block dimensions: " << ns_block.x << std::endl;
    std::cout << "Namespace grid dimensions: " << ns_grid.x << std::endl;
    
    std::cout << "CudaUtils.h syntax test passed!" << std::endl;
    return 0;
}
