// RandomStateInit.cu - Kernel implementation only
#include "../../include/NeuroGen/cuda/RandomStateInit.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

__global__ void initRandomStates(curandState* states, int num_states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_states) return;
    curand_init(seed, idx, 0, &states[idx]);
}


__global__ void initializeRandomStates(curandState* states, unsigned long seed, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Each thread gets same seed, different sequence number
        curand_init(seed, idx, 0, &states[idx]);
    }
}