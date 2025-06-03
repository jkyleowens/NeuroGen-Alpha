// SynapseInputKernel.cu — Fixed implementation file
#include <NeuroGen/cuda/CudaCompatibility.h>
#include "../../include/NeuroGen/cuda/SynapseInputKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/GridBlockUtils.cuh"
#include <cuda_runtime.h>

__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Check if presynaptic neuron spiked
    if (neurons[pre_idx].spiked) {
        // Record spike time for STDP
        synapse.last_pre_spike_time = neurons[pre_idx].last_spike_time;
        
        // Update activity metric
        synapse.activity_metric = synapse.activity_metric * 0.99f + 0.01f;
        
        // Apply synaptic input to postsynaptic neuron
        int compartment = synapse.post_compartment;
        int receptor = synapse.receptor_index;
        
        // Ensure indices are valid
        if (compartment >= 0 && compartment < MAX_COMPARTMENTS &&
            receptor >= 0 && receptor < MAX_SYNAPTIC_RECEPTORS) {
            
            // Add synaptic conductance (simplified model)
            atomicAdd(&neurons[post_idx].receptor_conductances[compartment][receptor], 
                     synapse.weight);
        }
    }
}