// Mock CUDA headers for testing
#ifndef MOCK_CUDA_RUNTIME_H
#define MOCK_CUDA_RUNTIME_H

#include <cstdlib>
#include <cstring>

// Mock CUDA types and functions
typedef int cudaError_t;
typedef int cudaStream_t;
typedef struct {} curandState;

#define cudaSuccess 0
#define __global__
#define __device__
#define __host__

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

// Mock CUDA functions
inline cudaError_t cudaMalloc(void** ptr, size_t size) { 
    *ptr = malloc(size); 
    return cudaSuccess; 
}
inline cudaError_t cudaFree(void* ptr) { 
    free(ptr); 
    return cudaSuccess; 
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) { 
    memcpy(dst, src, count); 
    return cudaSuccess; 
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) { return "Mock CUDA error"; }

// Mock memory copy kinds
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

#endif // MOCK_CUDA_RUNTIME_H
