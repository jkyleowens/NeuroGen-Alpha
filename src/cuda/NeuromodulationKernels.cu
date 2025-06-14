#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/LearningRuleConstants.h>

/**
 * @file NeuromodulationKernels.cu
 * @brief Implements kernels for applying neuromodulatory effects on the network.
 */

/**
 * @brief Applies neuromodulatory effects to individual neurons.
 *
 * This kernel adjusts intrinsic neuronal properties based on local neuromodulator levels.
 * For example, acetylcholine can increase excitability, while serotonin can decrease it.
 *
 * @param neurons Device pointer to neuron states.
 * @param global_modulators Device pointer to global neuromodulator levels [DA, ACh, 5-HT, NE].
 * @param num_neurons Total number of neurons.
 */
__global__ void applyNeuromodulationToNeuronsKernel(GPUNeuronState* neurons, const float* global_modulators, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];
    if (neuron.active == 0) return;

    // Unpack global modulator levels
    float ACh_level = global_modulators[1]; // Acetylcholine
    float SER_level = global_modulators[2]; // Serotonin

    // --- Apply Acetylcholine (ACh) Effects: Increased Attentiveness/Excitability ---
    // We model this as a slight depolarization, making the neuron easier to fire.
    float ach_effect = ACh_level * ACETYLCHOLINE_EXCITABILITY_FACTOR;

    // --- Apply Serotonin (5-HT) Effects: General Inhibition / Mood Stability ---
    // We model this as a slight hyperpolarization, making the neuron harder to fire.
    float ser_effect = SER_level * SEROTONIN_INHIBITORY_FACTOR;
    
    // The final excitability modulation is the balance of these effects
    neuron.neuromod_excitability = ach_effect - ser_effect;
    
    // This excitability factor can be used in the neuron update kernel to shift the resting potential
    // or adjust the spike threshold, dynamically altering network-wide activity.
    // For now, we store it. Integration into the neuron update kernel is a future step.
}

/**
 * @brief Applies neuromodulatory effects to individual synapses.
 *
 * This kernel adjusts synaptic properties, primarily the learning rate (plasticity).
 * Acetylcholine, associated with attention and learning, will increase the rate of plasticity.
 *
 * @param synapses Device pointer to synapse states.
 * @param global_modulators Device pointer to global neuromodulator levels.
 * @param num_synapses Total number of synapses.
 */
__global__ void applyNeuromodulationToSynapsesKernel(GPUSynapse* synapses, const float* global_modulators, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // Unpack global modulator levels
    float ACh_level = global_modulators[1]; // Acetylcholine
    
    // --- Acetylcholine (ACh) enhances plasticity ---
    // This simulates a state of high attention, where the brain is more receptive to learning.
    // The final learning rate in the STDP kernel will be multiplied by this factor.
    float ach_plasticity_bonus = ACh_level * synapse.acetylcholine_sensitivity * ACETYLCHOLINE_PLASTICITY_FACTOR;

    // Set the final plasticity modulation factor for this synapse
    // We start with a baseline of 1.0 and add the bonus.
    synapse.plasticity_modulation = 1.0f + ach_plasticity_bonus;
}