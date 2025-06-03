// CUDA Compatibility and type trait fixes
#include <NeuroGen/cuda/CudaCompatibility.h>
#include <NeuroGen/cuda/CudaUtils.h>

#include "../../include/NeuroGen/cuda/KernelLaunchWrappers.cuh"
#include "../../include/NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include "../../include/NeuroGen/cuda/NeuronSpikingKernels.cuh"
#include "../../include/NeuroGen/cuda/SynapseInputKernel.cuh"
#include "../../include/NeuroGen/cuda/STDPKernel.cuh"
#include "../../include/NeuroGen/cuda/RandomStateInit.cuh"
#include "../../include/NeuroGen/cuda/GridBlockUtils.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

extern "C" void launchUpdateNeuronVoltages(GPUNeuronState* neurons, float* I_leak, float* Cm, float dt, int N) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(N);
    updateNeuronVoltages<<<grid, block>>>(neurons, I_leak, Cm, dt, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchNeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

extern "C" void launchSynapseInputKernel(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_synapses);
    synapseInputKernel<<<grid, block>>>(d_synapses, d_neurons, num_synapses);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchSpikeDetectionKernel(GPUNeuronState* neurons, GPUSpikeEvent* spikes, float threshold,
                                int* spike_count, int num_neurons, float current_time) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(num_neurons, 256);
    detectSpikes<<<grid, block>>>(neurons, spikes, threshold, spike_count, num_neurons, current_time);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}