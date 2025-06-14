#include <NeuroGen/cuda/StructuralPlasticityKernels.cuh>
#include <NeuroGen/LearningRuleConstants.h>
#include <curand_kernel.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>

/**
 * @brief Marks weak or underutilized synapses for pruning.
 *
 * This kernel iterates through synapses and deactivates those that are
 * structurally weak (low weight) and functionally inactive (low activity metric).
 * This mimics the biological process of use-dependent synaptic pruning.
 */
__global__ void markPrunableSynapsesKernel(GPUSynapse* synapses, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // Pruning criteria: The synapse must be both weak and inactive.
    bool is_weak = fabsf(synapse.weight) < (MIN_WEIGHT / 10.0f);
    bool is_inactive = synapse.activity_metric < 0.01f;

    if (is_weak && is_inactive) {
        // Deactivate the synapse. In a more complex system, this could
        // free up the memory for a new synapse to form.
        synapse.active = 0;
    }
}


/**
 * @brief Drives neurogenesis by activating new neurons in "computationally stressed" regions.
 *
 * This kernel simulates neurogenesis by finding inactive neurons in the neuron pool
 * and activating them. The trigger is based on the overall network activity and
 * plasticity levels, simulating the biological principle that new neurons are
 * generated and recruited in response to challenging and dynamic environments.
 */
__global__ void neurogenesisKernel(GPUNeuronState* neurons, const GPUSynapse* synapses,
                                 curandState* rng_states, int num_neurons) {
    // This kernel is best executed with a single thread block to avoid race conditions
    // when finding an inactive neuron to activate.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        // --- 1. Assess the "Need" for New Neurons ---
        // As a proxy for network stress, we can check the average synaptic weight.
        // If weights are saturating at their maximum, it means the network is
        // struggling to represent new information and needs more capacity.
        float avg_weight = 0.0f;
        int active_synapses = 0;
        for (int i = 0; i < 5000; ++i) { // Sample a subset of synapses for performance
            int s_idx = (int)(curand_uniform(&rng_states[0]) * num_neurons);
            if(synapses[s_idx].active){
                avg_weight += fabsf(synapses[s_idx].weight);
                active_synapses++;
            }
        }
        if(active_synapses > 0) avg_weight /= active_synapses;

        // Trigger neurogenesis if the average weight is high, indicating saturation.
        if (avg_weight < (MAX_WEIGHT * 0.75f)) {
            return; // No need for new neurons yet.
        }

        // --- 2. Find an Inactive Neuron to Activate ---
        int new_neuron_idx = -1;
        for (int i = 0; i < num_neurons; ++i) {
            if (neurons[i].active == 0) {
                new_neuron_idx = i;
                break;
            }
        }

        // --- 3. Activate and Integrate the New Neuron ---
        if (new_neuron_idx != -1) {
            GPUNeuronState& new_neuron = neurons[new_neuron_idx];
            new_neuron.active = 1;
            new_neuron.voltage = V_REST;
            // Reset all state variables to their defaults
            // (A full implementation would be more detailed here)
            
            // This newly "born" neuron is now available for connection
            // via synaptogenesis in subsequent steps.
        }
    }
}