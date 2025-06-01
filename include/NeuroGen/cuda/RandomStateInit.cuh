#pragma once
#ifndef RANDOM_STATE_INIT_CUH
#define RANDOM_STATE_INIT_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Kernel declaration only - implementation in RandomStateInit.cu
__global__ void initializeRandomStates(curandState* states, unsigned long seed, int count);

#endif // RANDOM_STATE_INIT_CUH