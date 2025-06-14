#include <NeuroGen/cuda/NeuronInitialization.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/LearningRuleConstants.h>
#include <curand_kernel.h>

/**
 * @file NeuronInitialization.cu
 * @brief CUDA kernels for initializing neuron states on the GPU.
 * This file contains the logic to set up default values for all state
 * variables within the GPUNeuronState struct, ensuring a consistent starting
 * state for the network simulation.
 */

/**
 * @brief Initializes a single compartment within a neuron with default values.
 *
 * This helper function is called by the main initialization kernel to set up each
 * compartment (e.g., soma, dendrite) with its appropriate biophysical properties.
 *
 * @param neuron A reference to the GPUNeuronState object being initialized.
 * @param c The index of the compartment to initialize.
 * @param type The type of the compartment (e.g., COMPARTMENT_SOMA).
 * @param parent The index of the parent compartment (-1 for soma).
 * @param gen A reference to the random number generator for stochastic initialization.
 */
__device__ void initializeCompartment(GPUNeuronState& neuron, int c, int type, int parent, curandState* rng_state) {
    neuron.compartment_types[c] = type;
    neuron.parent_compartment[c] = parent;
    neuron.coupling_conductance[c] = 0.1f; // Default coupling

    // Initialize Hodgkin-Huxley and other channel states
    neuron.voltages[c] = V_REST;
    neuron.m_comp[c] = 0.05f; // Resting state for Na+ activation
    neuron.h_comp[c] = 0.6f;  // Resting state for Na+ inactivation
    neuron.n_comp[c] = 0.32f; // Resting state for K+ activation

    // Initialize calcium and other ion channel states within the 'channels' substruct
    neuron.channels.ampa_g[c] = 0.0f;
    neuron.channels.nmda_g[c] = 0.0f;
    neuron.channels.gaba_a_g[c] = 0.0f;
    neuron.channels.gaba_b_g[c] = 0.0f;
    neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;

    // Initialize dendritic spike properties
    neuron.dendritic_spike[c] = false;
    neuron.dendritic_spike_time[c] = -1e9f; // Far in the past
    neuron.dendritic_threshold[c] = -40.0f;   // Default dendritic spike threshold
}


/**
 * @brief CUDA kernel to initialize the state of all neurons in the network.
 *
 * This kernel sets all neurons to a default resting state at the beginning
 * of a simulation. It iterates through each neuron assigned to a thread and
 * configures its basic properties, compartment structure, and state variables.
 */
__global__ void initializeDefaultNeuronState(GPUNeuronState* neurons, int num_neurons, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    curandState rng_state;
    curand_init(seed, idx, 0, &rng_state);

    GPUNeuronState& neuron = neurons[idx];

    // --- Basic Properties ---
    neuron.neuron_id = idx;
    neuron.active = 1;
    // Simple 80/20 split for excitatory/inhibitory types
    neuron.type = (curand_uniform(&rng_state) < 0.8f) ? NEURON_EXCITATORY : NEURON_INHIBITORY;

    // --- Soma and Main HH Variables ---
    neuron.voltage = V_REST; // Main somatic voltage
    neuron.m = 0.05f;
    neuron.h = 0.6f;
    neuron.n = 0.32f;

    // --- Spiking Properties ---
    neuron.spiked = false;
    neuron.spike_count = 0;
    neuron.last_spike_time = -1e9f;
    neuron.spike_threshold = V_THRESH;
    neuron.refractory_period = 2.0f; // 2 ms refractory period

    // --- Activity and Homeostasis ---
    neuron.activity_level = 0.0f;
    neuron.average_activity = 0.0f;
    neuron.average_firing_rate = 0.0f;
    neuron.homeostatic_scaling_factor = 1.0f;
    neuron.adaptation_current = 0.0f;
    neuron.leak_conductance = HH_G_L; // from NeuronModelConstants.h
    neuron.homeostatic_target = TARGET_ACTIVITY_LEVEL; // from LearningRuleConstants.h

    // --- Neuromodulation States ---
    neuron.dopamine_level = 0.0f;
    neuron.serotonin_level = 0.0f;
    neuron.acetylcholine_level = 0.0f;
    neuron.noradrenaline_level = 0.0f;
    neuron.neuromod_excitability = 0.0f;
    neuron.neuromod_ampa_scale = 1.0f;
    neuron.neuromod_nmda_scale = 1.0f;
    neuron.neuromod_gaba_scale = 1.0f;
    neuron.neuromod_adaptation = 1.0f;
    
    // --- Compartment Initialization (Example: Soma + 3 Dendritic compartments) ---
    neuron.compartment_count = 4;
    initializeCompartment(neuron, 0, COMPARTMENT_SOMA, -1, &rng_state);
    initializeCompartment(neuron, 1, COMPARTMENT_BASAL, 0, &rng_state);
    initializeCompartment(neuron, 2, COMPARTMENT_APICAL, 0, &rng_state);
    initializeCompartment(neuron, 3, COMPARTMENT_APICAL, 2, &rng_state);

    // Initialize remaining unused compartments to inactive
    for (int c = neuron.compartment_count; c < MAX_COMPARTMENTS; ++c) {
        neuron.compartment_types[c] = COMPARTMENT_INACTIVE;
    }
}