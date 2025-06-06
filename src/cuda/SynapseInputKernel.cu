#include <NeuroGen/cuda/SynapseInputKernel.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronModelConstants.h> // For receptor type definitions
#include <cuda_runtime.h>

/**
 * @file SynapseInputKernel.cu
 * @brief CUDA kernel for processing synaptic inputs with receptor specificity.
 *
 * This kernel has been significantly updated to be compatible with the new ion
 * channel dynamics introduced in the enhanced neuron model. When a presynaptic
 * neuron fires, this kernel no longer injects a generic current. Instead, it
 * delivers the synaptic weight to the appropriate receptor's state variable
 * on the postsynaptic neuron's target compartment. This state variable is then
 * used by the `enhancedRK4NeuronUpdateKernel` to calculate the actual current
 * based on the receptor's specific kinetics (e.g., rise and decay times).
 *
 * Key Enhancements:
 * 1.  **Receptor-Specific Targeting**: Uses `atomicAdd` to update the correct
 * receptor state variable (e.g., `ampa_state`, `nmda_state`) based on
 * the synapse's properties.
 * 2.  **Direct State-Variable Interaction**: Provides the necessary input to the
 * dual-exponential synapse models now running in the neuron update kernel.
 * 3.  **Preserves Spike Timing**: Correctly updates the `last_pre_spike_time`
 * on the synapse, which is critical for the future implementation of
 * spike-timing-dependent plasticity (STDP).
 */
__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];

    // Skip inactive synapses
    if (synapse.active == 0) return;

    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;

    // Proceed only if the presynaptic neuron has fired in the current step
    if (neurons[pre_idx].spiked) {
        // Record the time of the presynaptic spike for plasticity calculations
        synapse.last_pre_spike_time = neurons[pre_idx].last_spike_time;

        // Update a simple activity metric for the synapse
        synapse.activity_metric = synapse.activity_metric * 0.99f + 0.01f;

        // Get the target compartment and receptor on the postsynaptic neuron
        int compartment = synapse.post_compartment;
        int receptor = synapse.receptor_index;

        // Ensure target indices are within valid bounds
        if (compartment >= 0 && compartment < MAX_COMPARTMENTS &&
            receptor >= 0 && receptor < NUM_RECEPTOR_TYPES) { // Assumes NUM_RECEPTOR_TYPES is defined

            // Atomically add the synaptic weight to the appropriate receptor's state variable.
            // This 'state' variable feeds the kinetic model in the neuron update kernel.
            // It represents the concentration of neurotransmitter ready to open channels.
            switch (receptor) {
                case RECEPTOR_AMPA:
                    atomicAdd(&neurons[post_idx].channels.ampa_state[compartment], synapse.weight);
                    break;
                case RECEPTOR_NMDA:
                    atomicAdd(&neurons[post_idx].channels.nmda_state[compartment], synapse.weight);
                    break;
                case RECEPTOR_GABA_A:
                    // Inhibitory weights are negative, so we add them directly.
                    atomicAdd(&neurons[post_idx].channels.gaba_a_state[compartment], synapse.weight);
                    break;
                case RECEPTOR_GABA_B:
                    atomicAdd(&neurons[post_idx].channels.gaba_b_state[compartment], synapse.weight);
                    break;
            }
        }
    }
}