#pragma once
#ifndef NETWORK_UPDATE_STUB_H
#define NETWORK_UPDATE_STUB_H

#include <cuda_runtime.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/NeuronInitialization.cuh>

static inline void initializeNeuronArray(GPUNeuronState** d_neurons, int N) {
    cudaMalloc(reinterpret_cast<void**>(d_neurons), N * sizeof(GPUNeuronState));
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    initializeNeuronCompartments<<< gridSize, blockSize >>>(*d_neurons, N);
    cudaDeviceSynchronize();
}

static inline void allocateDeviceMemory(float** d_mem, int N) {
    cudaMalloc(reinterpret_cast<void**>(d_mem), N * sizeof(float));
    cudaMemset(*d_mem, 0, N * sizeof(float));
}

static inline void cleanupCudaMemory(GPUNeuronState* d_neurons, void* /*unused1*/, void* /*unused2*/) {
    cudaFree(d_neurons);
}

class NetworkCUDA {
public:
    NetworkCUDA(int num_neurons)
        : num_neurons_(num_neurons), d_neurons_(nullptr), d_currents_(nullptr) {
        initializeNeuronArray(&d_neurons_, num_neurons_);
        allocateDeviceMemory(&d_currents_, num_neurons_);
    }

    ~NetworkCUDA() {
        cleanupCudaMemory(d_neurons_, nullptr, nullptr);
        cudaFree(d_currents_);
    }

    void step(float dt, float current_time) {
        launchRK4NeuronUpdateKernel(d_neurons_, num_neurons_, dt, current_time);
    }

    GPUNeuronState* getDeviceNeurons() const { return d_neurons_; }
    float* getDeviceCurrents() const { return d_currents_; }

private:
    int num_neurons_;
    GPUNeuronState* d_neurons_;
    float* d_currents_;
};

#endif // NETWORK_UPDATE_STUB_H
