#include "../../include/NeuroGen/cuda/NeuronSpikingKernels.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include <cuda_runtime.h>

__global__ void countSpikesKernel(const GPUNeuronState* neurons,
                                 int* spike_count, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    if (neurons[idx].spiked) {
        atomicAdd(spike_count, 1);
    }
}

__global__ void updateNeuronSpikes(GPUNeuronState* neurons,
                                  float threshold, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Check for spike
    if (neuron.voltage >= threshold && !neuron.spiked) {
        neuron.spiked = true;
    } else {
        neuron.spiked = false;
    }
}

__global__ void detectSpikes(const GPUNeuronState* neurons,
                            GPUSpikeEvent* spikes, float threshold,
                            int* spike_count, int num_neurons, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = neurons[idx];
    
    // Check for spike
    if (neuron.voltage >= threshold) {
        // Record spike event
        int spike_idx = atomicAdd(spike_count, 1);
        
        // Ensure we don't overflow the spike buffer
        if (spike_idx < num_neurons * 10) {
            spikes[spike_idx].neuron_idx = idx;
            spikes[spike_idx].time = current_time;
            spikes[spike_idx].amplitude = 1.0f;
        }
    }
}