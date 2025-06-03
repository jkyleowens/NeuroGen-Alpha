#include <NeuroGen/cuda/CudaCompatibility.h>
#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel to test compilation
__global__ void testKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

int main() {
    std::cout << "Testing CUDA compilation fixes..." << std::endl;
    
    // Check CUDA device
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cout << "CUDA not available: " << cudaGetErrorString(err) << std::endl;
        return 0; // Not an error, just no CUDA
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    const int N = 1024;
    float* h_data = new float[N];
    float* d_data;
    
    // Allocate device memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_data, N * sizeof(float)));
    
    // Launch kernel
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    testKernel<<<grid, block>>>(d_data, N);
    
    // Check for kernel errors
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_SYNC_CHECK();
    
    // Copy back
    CUDA_CHECK_ERROR(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        if (h_data[i] != i * 2.0f) {
            success = false;
            break;
        }
    }
    
    std::cout << "CUDA compilation test: " << (success ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
    
    return success ? 0 : 1;
}
