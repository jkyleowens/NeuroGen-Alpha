// Test file to check CUDA type compatibility
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

__global__ void testKernel() {
    dim3 block(256);
    dim3 grid(1);
    float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    int4 i4 = make_int4(1, 2, 3, 4);
    cudaError_t err = cudaSuccess;
}

int main() {
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
