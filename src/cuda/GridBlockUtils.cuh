#pragma once
#include <cuda_runtime.h>

#define DEFAULT_BLOCK_SIZE 256

__host__ __device__ inline dim3 makeBlock() {
    return dim3(DEFAULT_BLOCK_SIZE);
}

__host__ __device__ inline dim3 makeGrid(int N) {
    return dim3((N + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
}
