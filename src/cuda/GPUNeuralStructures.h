#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include <cuda_runtime.h>

// Include other GPU structures
#include "CorticalColumn.h"
#include "NeuronModelConstants.h"

// Note: MAX_COMPARTMENTS and MAX_SYNAPTIC_RECEPTORS are now defined in NeuronModelConstants.h

/**
 * GPU-optimized structure for neuron state
 */
struct GPUNeuronState {
    // Soma properties
    float voltage;                     // Somatic membrane potential
    float m;                           // Activation variable for sodium channel
    float h;                           // Inactivation variable for sodium channel
    float n;                           // Activation variable for potassium channel
    float I_ext;                       // External current input
    float x, y, z;                     // 3D position coordinates
    int neuron_type;                   // Neuron type (e.g., excitatory, inhibitory, direct pathway, etc.)
    bool spiked;                       // Whether the neuron has spiked in the current timestep
    int active;                        // Whether the neuron is active (1) or inactive (0)
    float last_spike_time;             // Time of the last spike
    float activity_level;              // Measure of recent activity
    
    // Compartment properties
    int compartment_count;             // Number of compartments
    int compartment_types[MAX_COMPARTMENTS]; // Type of each compartment (basal, apical, etc.)
    float voltages[MAX_COMPARTMENTS];  // Voltages for each compartment
    
    // Ion channel states for each compartment
    float m_comp[MAX_COMPARTMENTS];    // Na activation for each compartment
    float h_comp[MAX_COMPARTMENTS];    // Na inactivation for each compartment
    float n_comp[MAX_COMPARTMENTS];    // K activation for each compartment
    
    // Calcium dynamics
    float ca_conc[MAX_COMPARTMENTS];           // Calcium concentration
    float ca_buffer[MAX_COMPARTMENTS];         // Calcium buffer concentration
    float ca_pump_rate[MAX_COMPARTMENTS];      // Calcium extrusion rate
    float ca_influx_modulation[MAX_COMPARTMENTS]; // Modulation of calcium influx
    
    // Ion channel states
    struct {
        // Synaptic channels
        float ampa_g[MAX_COMPARTMENTS];        // AMPA conductance
        float ampa_state[MAX_COMPARTMENTS];    // AMPA state variable
        float nmda_g[MAX_COMPARTMENTS];        // NMDA conductance
        float nmda_state[MAX_COMPARTMENTS];    // NMDA state variable
        float gaba_a_g[MAX_COMPARTMENTS];      // GABA-A conductance
        float gaba_a_state[MAX_COMPARTMENTS];  // GABA-A state variable
        float gaba_b_g[MAX_COMPARTMENTS];      // GABA-B conductance
        float gaba_b_state[MAX_COMPARTMENTS];  // GABA-B state variable
        float gaba_b_g_protein[MAX_COMPARTMENTS]; // GABA-B G-protein level
        
        // Voltage-gated channels
        float ca_m[MAX_COMPARTMENTS];          // Ca channel activation
        float kca_m[MAX_COMPARTMENTS];         // KCa channel activation
        float hcn_h[MAX_COMPARTMENTS];         // HCN channel activation
    } channels;
    
    // Membrane properties
    float I_leak[MAX_COMPARTMENTS];    // Leak currents for each compartment
    float Cm[MAX_COMPARTMENTS];        // Membrane capacitances for each compartment
    float resting_potential;           // Base resting potential
    float resting_potential_modulated; // Modulated resting potential
    float spike_threshold;             // Base spike threshold
    float spike_threshold_modulated;   // Modulated spike threshold
    float k_conductance_modulation;    // Modulation of K+ conductance (soma)
    float k_conductance_modulation_dendrites[MAX_COMPARTMENTS]; // K+ conductance modulation (dendrites)
    
    // Synaptic properties
    float receptor_conductances[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS]; // Synaptic receptor conductances
    float receptor_states[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS]; // Additional state variables
    
    // Dendritic spike properties
    bool dendritic_spike[MAX_COMPARTMENTS];  // Whether a dendritic spike occurred
    float last_dendritic_spike[MAX_COMPARTMENTS]; // Time of last dendritic spike
    
    // Compartment connectivity
    int parent_compartment[MAX_COMPARTMENTS]; // Parent compartment index (-1 for soma)
    float coupling_conductance[MAX_COMPARTMENTS]; // Conductance to parent
    
    // Neuromodulation-related fields (for Phase 4)
    struct {
        float dopamine;                // Local dopamine level
        float serotonin;               // Local serotonin level
        float acetylcholine;           // Local acetylcholine level
        float noradrenaline;           // Local noradrenaline level
        
        // Receptor desensitization state
        float dopamine_desensitization;
        float serotonin_desensitization;
        float acetylcholine_desensitization;
        float noradrenaline_desensitization;
    } neuromodulators;
    
    // Neuromodulator sensitivity
    float dopamine_sensitivity;
    float serotonin_sensitivity;
    float acetylcholine_sensitivity;
    float noradrenaline_sensitivity;
    
    // State regulation
    float excitability_modulation;    // Current modulation of excitability
    float plasticity_modulation;      // Current modulation of plasticity
    float adaptation_rate;            // Rate of adaptation to neuromodulators
};

/**
 * GPU-optimized structure for synapses
 */
struct GPUSynapse {
    int pre_neuron;            // ID of the presynaptic neuron (legacy field)
    int post_neuron;           // ID of the postsynaptic neuron (legacy field)
    int pre_neuron_idx;        // Index of the presynaptic neuron
    int post_neuron_idx;       // Index of the postsynaptic neuron
    float weight;              // Synaptic weight
    float effective_weight;    // Modulated effective weight
    float delay;               // Synaptic delay in milliseconds
    float last_active;         // Time of last activation
    int type;                  // Synapse type
    int active;                // Whether the synapse is active (1) or inactive (0)
    float last_pre_spike_time; // Time of last presynaptic spike
    float activity_metric;     // Metric of recent activity
    float last_potentiation;   // Time of last potentiation
    int post_compartment;      // Target compartment on postsynaptic neuron
    int receptor_index;        // Target receptor type
    
    // Enhanced eligibility trace system (for Phase 3)
    float eligibility_trace;   // Legacy eligibility trace field
    float fast_trace;          // Fast eligibility trace (tens of ms)
    float medium_trace;        // Medium eligibility trace (seconds)
    float slow_trace;          // Slow eligibility trace (minutes)
    float tag_strength;        // Synaptic tag for late-phase plasticity
    
    // Plasticity parameters
    float plasticity_rate;     // Learning rate for this synapse
    float meta_weight;         // Metaplastic weight (controls plasticity threshold)
    float recent_activity;     // Measure of recent activity
};

/**
 * GPU-optimized structure for spike events
 */
struct GPUSpikeEvent {
    int neuron_idx;    // Index of the neuron that spiked
    float time;        // Time of the spike
    float amplitude;   // Amplitude of the spike
};

#endif // GPU_NEURAL_STRUCTURES_H
