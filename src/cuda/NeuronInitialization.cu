#include "NeuronInitialization.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "NeuronModelConstants.h"
#include <cuda_runtime.h>

/**
 * CUDA kernel for initializing neuron compartments
 * This function sets up the multi-compartment structure for each neuron
 */
__global__ void initializeNeuronCompartments(GPUNeuronState* neurons, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Initialize soma (compartment 0)
    neuron.voltages[0] = RESTING_POTENTIAL;
    neuron.m_comp[0] = 0.05f;
    neuron.h_comp[0] = 0.6f;
    neuron.n_comp[0] = 0.32f;
    neuron.ca_conc[0] = RESTING_CA_CONCENTRATION;
    neuron.ca_buffer[0] = 0.0f;
    neuron.ca_pump_rate[0] = 0.2f; // Faster in soma
    neuron.ca_influx_modulation[0] = 1.0f;
    neuron.compartment_types[0] = COMPARTMENT_SOMA;
    neuron.parent_compartment[0] = -1; // No parent for soma
    
    // Initialize resting and threshold potentials
    neuron.resting_potential = RESTING_POTENTIAL;
    neuron.resting_potential_modulated = RESTING_POTENTIAL;
    neuron.spike_threshold = SPIKE_THRESHOLD;
    neuron.spike_threshold_modulated = SPIKE_THRESHOLD;
    
    // Initialize K+ conductance modulation
    neuron.k_conductance_modulation = 1.0f;
    
    // Initialize activity level
    neuron.activity_level = 0.0f;
    
    // Initialize neuromodulator sensitivities
    neuron.dopamine_sensitivity = 1.0f;
    neuron.serotonin_sensitivity = 1.0f;
    neuron.acetylcholine_sensitivity = 1.0f;
    neuron.noradrenaline_sensitivity = 1.0f;
    
    // Initialize neuromodulator levels
    neuron.neuromodulators.dopamine = 0.0f;
    neuron.neuromodulators.serotonin = 0.0f;
    neuron.neuromodulators.acetylcholine = 0.0f;
    neuron.neuromodulators.noradrenaline = 0.0f;
    
    // Initialize neuromodulator desensitization
    neuron.neuromodulators.dopamine_desensitization = 0.0f;
    neuron.neuromodulators.serotonin_desensitization = 0.0f;
    neuron.neuromodulators.acetylcholine_desensitization = 0.0f;
    neuron.neuromodulators.noradrenaline_desensitization = 0.0f;
    
    // Initialize state regulation
    neuron.excitability_modulation = 0.0f;
    neuron.plasticity_modulation = 0.0f;
    neuron.adaptation_rate = 0.01f;
    
    // Initialize ion channel states
    for (int c = 0; c < MAX_COMPARTMENTS; c++) {
        neuron.channels.ampa_g[c] = 0.0f;
        neuron.channels.ampa_state[c] = 0.0f;
        neuron.channels.nmda_g[c] = 0.0f;
        neuron.channels.nmda_state[c] = 0.0f;
        neuron.channels.gaba_a_g[c] = 0.0f;
        neuron.channels.gaba_a_state[c] = 0.0f;
        neuron.channels.gaba_b_g[c] = 0.0f;
        neuron.channels.gaba_b_state[c] = 0.0f;
        neuron.channels.gaba_b_g_protein[c] = 0.0f;
        
        neuron.channels.ca_m[c] = 0.05f;
        neuron.channels.kca_m[c] = 0.05f;
        neuron.channels.hcn_h[c] = 0.05f;
        
        neuron.k_conductance_modulation_dendrites[c] = 1.0f;
    }
    
    // Initialize dendritic compartments based on neuron type
    if (neuron.neuron_type == NEURON_DIRECT_PATHWAY || 
        neuron.neuron_type == NEURON_REWARD_PREDICTION) {
        // Complex neuron with both basal and apical dendrites
        neuron.compartment_count = 5;
        
        // Basal dendrites (compartments 1-2)
        for (int c = 1; c <= 2; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.ca_buffer[c] = 0.0f;
            neuron.ca_pump_rate[c] = 0.15f; // Intermediate in basal dendrites
            neuron.ca_influx_modulation[c] = 1.0f;
            neuron.compartment_types[c] = COMPARTMENT_BASAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = BASAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
            neuron.last_dendritic_spike[c] = -1000.0f;
        }
        
        // Apical dendrites (compartments 3-4)
        for (int c = 3; c <= 4; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.ca_buffer[c] = 0.0f;
            neuron.ca_pump_rate[c] = 0.1f; // Slower in apical dendrites
            neuron.ca_influx_modulation[c] = 1.0f;
            neuron.compartment_types[c] = COMPARTMENT_APICAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = APICAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
            neuron.last_dendritic_spike[c] = -1000.0f;
        }
    } else {
        // Simple neuron with just basal dendrites
        neuron.compartment_count = 3;
        
        // Basal dendrites (compartments 1-2)
        for (int c = 1; c <= 2; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.ca_buffer[c] = 0.0f;
            neuron.ca_pump_rate[c] = 0.15f;
            neuron.ca_influx_modulation[c] = 1.0f;
            neuron.compartment_types[c] = COMPARTMENT_BASAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = BASAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
            neuron.last_dendritic_spike[c] = -1000.0f;
        }
    }
    
    // Initialize receptor conductances
    for (int c = 0; c < neuron.compartment_count; c++) {
        for (int r = 0; r < MAX_SYNAPTIC_RECEPTORS; r++) {
            neuron.receptor_conductances[c][r] = 0.0f;
            neuron.receptor_states[c][r] = 0.0f;
        }
    }
}

/**
 * CUDA kernel for resetting neuron state
 * This function resets the dynamic state of neurons without changing their structure
 */
__global__ void resetNeuronState(GPUNeuronState* neurons, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Reset spike state
    neuron.spiked = false;
    neuron.last_spike_time = -1000.0f;
    
    // Reset voltages
    neuron.voltage = RESTING_POTENTIAL;
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.voltages[c] = RESTING_POTENTIAL;
    }
    
    // Reset gating variables
    neuron.m = 0.05f;
    neuron.h = 0.6f;
    neuron.n = 0.32f;
    
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.m_comp[c] = 0.05f;
        neuron.h_comp[c] = 0.6f;
        neuron.n_comp[c] = 0.32f;
    }
    
    // Reset dendritic spike state
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.dendritic_spike[c] = false;
        neuron.last_dendritic_spike[c] = -1000.0f;
    }
    
    // Reset receptor conductances
    for (int c = 0; c < neuron.compartment_count; c++) {
        for (int r = 0; r < MAX_SYNAPTIC_RECEPTORS; r++) {
            neuron.receptor_conductances[c][r] = 0.0f;
            neuron.receptor_states[c][r] = 0.0f;
        }
    }
    
    // Reset ion channel states
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.channels.ampa_g[c] = 0.0f;
        neuron.channels.ampa_state[c] = 0.0f;
        neuron.channels.nmda_g[c] = 0.0f;
        neuron.channels.nmda_state[c] = 0.0f;
        neuron.channels.gaba_a_g[c] = 0.0f;
        neuron.channels.gaba_a_state[c] = 0.0f;
        neuron.channels.gaba_b_g[c] = 0.0f;
        neuron.channels.gaba_b_state[c] = 0.0f;
        neuron.channels.gaba_b_g_protein[c] = 0.0f;
        
        neuron.channels.ca_m[c] = 0.05f;
        neuron.channels.kca_m[c] = 0.05f;
        neuron.channels.hcn_h[c] = 0.05f;
    }
    
    // Reset calcium concentrations
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
        neuron.ca_buffer[c] = 0.0f;
    }
    
    // Reset neuromodulator-related state
    neuron.excitability_modulation = 0.0f;
    neuron.plasticity_modulation = 0.0f;
    
    // Don't reset structure or parameters
}
