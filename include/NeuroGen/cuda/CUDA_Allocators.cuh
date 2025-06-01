// Extended CUDA allocators for neurons, synapses, and spike events
#ifndef CUDA_ALLOCATORS_EXTENDED_H
#define CUDA_ALLOCATORS_EXTENDED_H

#include <cuda_runtime.h>
#include <cstdio>
#include "GPUNeuralStructures.h"

#include <curand_kernel.h> // Required for curandState


// Template allocation helper
template <typename T>
void allocateDeviceMemory(T** ptr, size_t count) {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        *ptr = nullptr;
    }
}

template <typename T>
void freeDeviceMemory(T* ptr) {
    cudaFree(ptr);
}

// Allocate and initialize neuron array
inline void initializeNeuronArray(GPUNeuronState** d_neurons, int num_neurons) {
    GPUNeuronState* temp;
    allocateDeviceMemory(&temp, num_neurons);
    cudaMemset(temp, 0, sizeof(GPUNeuronState) * num_neurons);
    *d_neurons = temp;
}

// Allocate and initialize synapse array
inline void initializeSynapseArray(GPUSynapse** d_synapses, int num_synapses) {
    GPUSynapse* temp;
    allocateDeviceMemory(&temp, num_synapses);
    cudaMemset(temp, 0, sizeof(GPUSynapse) * num_synapses);
    *d_synapses = temp;
}

// Allocate and initialize spike event buffer
inline void initializeSpikeBuffer(GPUSpikeEvent** d_spikes, int max_spikes) {
    GPUSpikeEvent* temp;
    allocateDeviceMemory(&temp, max_spikes);
    cudaMemset(temp, 0, sizeof(GPUSpikeEvent) * max_spikes);
    *d_spikes = temp;
}

// Cleanup routine for all major pointers
inline void cleanupCudaMemory(GPUNeuronState* d_neurons, GPUSynapse* d_synapses, GPUSpikeEvent* d_spikes) {
    if (d_neurons) cudaFree(d_neurons);
    if (d_synapses) cudaFree(d_synapses);
    if (d_spikes) cudaFree(d_spikes);
}

inline void allocateNeuronState(GPUNeuronState** ptr, int count) {
    cudaMalloc(ptr, sizeof(GPUNeuronState) * count);
}

inline void allocateInputCurrents(float** ptr, int count) {
    cudaMalloc(ptr, sizeof(float) * count);
    cudaMemset(*ptr, 0, sizeof(float) * count);
}

inline void allocateSpikeEvents(GPUSpikeEvent** ptr, int count) {
    cudaMalloc(ptr, sizeof(GPUSpikeEvent) * count);
}

inline void allocateSpikeCount(int** ptr) {
    cudaMalloc(ptr, sizeof(int));
    cudaMemset(*ptr, 0, sizeof(int));
}

inline void allocateRandomStates(curandState** ptr, int count) {
    cudaMalloc(ptr, sizeof(curandState) * count);
}



#endif // CUDA_ALLOCATORS_EXTENDED_H
