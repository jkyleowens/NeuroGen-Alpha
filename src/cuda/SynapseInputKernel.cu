// SynapseInputKernel.cu — Fixed implementation file
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/SynapseInputKernel.cuh"

__global__ void applySynapticCurrents(const GPUSynapse* synapses, 
                                     int num_synapses, 
                                     float* input_currents, 
                                     const GPUNeuronState* neurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_synapses) return;

    const GPUSynapse& syn = synapses[i];

    // Check if presynaptic neuron has spiked
    if (neurons[syn.pre_neuron_idx].spiked) {
        atomicAdd(&input_currents[syn.post_neuron_idx], syn.weight);
    }
}

__global__ void synapseInputKernel(GPUSynapse* synapses, 
                                  GPUNeuronState* neurons, 
                                  int num_synapses) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_synapses) {
        GPUSynapse& synapse = synapses[idx];
        int pre_idx = synapse.pre_neuron_idx;
        int post_idx = synapse.post_neuron_idx;
        
        // Check if presynaptic neuron has spiked
        if (neurons[pre_idx].spiked) {
            // Apply synaptic weight to postsynaptic neuron voltage
            atomicAdd(&neurons[post_idx].voltages[0], synapse.weight);
            
            // Update synapse activity for plasticity tracking
            synapse.activity_metric += 1.0f;
        }
    }
}