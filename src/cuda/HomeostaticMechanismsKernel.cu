#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/LearningRuleConstants.h>

/**
 * @file HomeostaticMechanismsKernel.cu
 * @brief Implements homeostatic plasticity rules to ensure network stability.
 */

/**
 * @brief Updates each neuron's average firing rate and computes a scaling
 * factor to apply to its incoming synapses.
 *
 * This kernel is the first step in synaptic scaling. It calculates how much a
 * neuron's synapses need to be scaled up or down to guide the neuron back to
 * its target firing rate. The computed factor is stored on the neuron.
 */
__global__ void computeSynapticScalingFactorKernel(GPUNeuronState* neurons, float dt, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];
    if (neuron.active == 0) return;

    // 1. Update the running average of the neuron's firing rate.
    // This uses an exponential moving average for efficiency.
    float decay_factor = expf(-dt / FIRING_RATE_TAU);
    float current_spike = neuron.spiked ? 1.0f / dt : 0.0f; // Instantaneous rate
    neuron.avg_firing_rate = neuron.avg_firing_rate * decay_factor + current_spike * (1.0f - decay_factor);

    // 2. Calculate the error between the current rate and the homeostatic target rate.
    float rate_error = neuron.avg_firing_rate - TARGET_FIRING_RATE;

    // 3. Compute the scaling factor. This is a slow, multiplicative adjustment.
    float scaling_adjustment = -rate_error * SYNAPTIC_SCALING_RATE * dt;
    float scaling_factor = 1.0f + scaling_adjustment;

    // Clamp the scaling factor to prevent extreme, destabilizing changes.
    neuron.homeostatic_scaling_factor = fmaxf(0.999f, fminf(1.001f, scaling_factor));
}

/**
 * @brief Applies the computed scaling factor to all synapses targeting each neuron.
 *
 * This kernel must be run after computeSynapticScalingFactorKernel. It iterates
 * through all synapses and adjusts their weights based on the scaling factor
ic
 * stored on the postsynaptic neuron.
 */
__global__ void applySynapticScalingKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // Get the scaling factor from the postsynaptic neuron.
    float scaling_factor = neurons[synapse.post_neuron_idx].homeostatic_scaling_factor;

    // Apply the scaling factor to the synaptic weight.
    synapse.weight *= scaling_factor;

    // Enforce absolute weight bounds.
    synapse.weight = fmaxf(MIN_WEIGHT, fminf(MAX_WEIGHT, synapse.weight));
}