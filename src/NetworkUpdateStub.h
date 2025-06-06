#pragma once
#ifndef NETWORK_UPDATE_STUB_H
#define NETWORK_UPDATE_STUB_H

#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>

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
        launchUpdateNeuronVoltages(d_neurons_, d_currents_, dt, current_time, num_neurons_);
    }

    GPUNeuronState* getDeviceNeurons() const { return d_neurons_; }
    float* getDeviceCurrents() const { return d_currents_; }

private:
    int num_neurons_;
    GPUNeuronState* d_neurons_;
    float* d_currents_;
};

#endif // NETWORK_UPDATE_STUB_H
