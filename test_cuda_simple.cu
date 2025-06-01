// Simple CUDA compilation test
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void testKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", idx);
}

int main() {
    std::cout << "Testing CUDA compilation..." << std::endl;
    
    // Check CUDA device
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
    
    // Launch simple kernel
    testKernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "CUDA test completed successfully!" << std::endl;
    return 0;
}
