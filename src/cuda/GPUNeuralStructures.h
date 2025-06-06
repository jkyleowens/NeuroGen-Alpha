#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include <cuda_runtime.h>
#include <NeuroGen/IonChannelConstants.h>

// Maximum number of compartments per neuron
#define MAX_COMPARTMENTS 8

/**
 * GPU-optimized structure for individual neurons with enhanced ion channel dynamics
 * Extended from Phase 1 to include comprehensive ion channel states
 */
struct GPUNeuronState {
    // ========================================
    // BASIC NEURON PROPERTIES (from Phase 1)
    // ========================================
    int neuron_id;                      // Unique identifier
    int active;                         // Whether neuron is active (1) or inactive (0)
    int type;                          // Neuron type (excitatory/inhibitory)
    
    // ========================================
    // COMPARTMENT STRUCTURE (from Phase 1)
    // ========================================
    int compartment_count;                              // Number of compartments
    int compartment_types[MAX_COMPARTMENTS];            // Type of each compartment
    int parent_compartment[MAX_COMPARTMENTS];           // Parent compartment index (-1 for soma)
    float coupling_conductance[MAX_COMPARTMENTS];       // Coupling conductance to parent
    
    // ========================================
    // VOLTAGE AND BASIC HH DYNAMICS (from Phase 1)
    // ========================================
    float voltage;                      // Soma membrane potential (mV)
    float voltages[MAX_COMPARTMENTS];   // Compartment voltages (mV)
    
    // Hodgkin-Huxley state variables for each compartment
    float m, h, n;                      // Soma HH variables
    float m_comp[MAX_COMPARTMENTS];     // Sodium activation
    float h_comp[MAX_COMPARTMENTS];     // Sodium inactivation  
    float n_comp[MAX_COMPARTMENTS];     // Potassium activation
    
    // ========================================
    // ION CHANNEL STATES (NEW IN PHASE 2)
    // ========================================
    struct {
        // Synaptic receptor states for each compartment
        float ampa_g[MAX_COMPARTMENTS];         // AMPA conductance (nS)
        float ampa_state[MAX_COMPARTMENTS];     // AMPA state variable
        float nmda_g[MAX_COMPARTMENTS];         // NMDA conductance (nS)
        float nmda_state[MAX_COMPARTMENTS];     // NMDA state variable
        float gaba_a_g[MAX_COMPARTMENTS];       // GABA-A conductance (nS)
        float gaba_a_state[MAX_COMPARTMENTS];   // GABA-A state variable
        float gaba_b_g[MAX_COMPARTMENTS];       // GABA-B conductance (nS)
        float gaba_b_state[MAX_COMPARTMENTS];   // GABA-B state variable
        float gaba_b_g_protein[MAX_COMPARTMENTS]; // GABA-B G-protein activation
        
        // Voltage-gated channel states for each compartment
        float ca_m[MAX_COMPARTMENTS];           // Ca channel activation
        float kca_m[MAX_COMPARTMENTS];          // KCa channel activation
        float hcn_h[MAX_COMPARTMENTS];          // HCN channel activation
    } channels;
    
    // ========================================
    // CALCIUM DYNAMICS (NEW IN PHASE 2)
    // ========================================
    float ca_conc[MAX_COMPARTMENTS];            // Calcium concentration (mM)
    float ca_buffer[MAX_COMPARTMENTS];          // Calcium buffer concentration
    float ca_pump_rate[MAX_COMPARTMENTS];       // Calcium extrusion rate (1/ms)
    
    // ========================================
    // LEGACY RECEPTOR CONDUCTANCES (from Phase 1)
    // ========================================
    // Keep for backward compatibility, gradually migrate to new channel states
    float receptor_conductances[MAX_COMPARTMENTS][NUM_RECEPTOR_TYPES];
    
    // ========================================
    // SPIKE GENERATION AND TIMING
    // ========================================
    bool spiked;                        // Whether neuron spiked this timestep
    float last_spike_time;              // Time of last spike (ms)
    float spike_threshold;              // Base spike threshold (mV)
    float spike_threshold_modulated;    // Modulated spike threshold (mV)
    float refractory_period;            // Refractory period duration (ms)
    float time_since_spike;             // Time since last spike (ms)
    
    // ========================================
    // ACTIVITY AND PLASTICITY METRICS
    // ========================================
    float activity_level;               // Recent activity level (0-1)
    float avg_firing_rate;              // Average firing rate (Hz)
    float membrane_resistance;          // Input resistance (MΩ)
    float membrane_capacitance;         // Membrane capacitance (pF)
    
    // ========================================
    // DENDRITIC PROCESSING (from Phase 1)
    // ========================================
    bool dendritic_spike[MAX_COMPARTMENTS];     // Dendritic spike flags
    float dendritic_spike_time[MAX_COMPARTMENTS]; // Time of last dendritic spike
    float dendritic_threshold[MAX_COMPARTMENTS];  // Dendritic spike thresholds
    
    // ========================================
    // SYNAPTIC INPUT TRACKING
    // ========================================
    float total_excitatory_input;       // Total excitatory synaptic current
    float total_inhibitory_input;       // Total inhibitory synaptic current
    float synaptic_input_rate;          // Rate of synaptic inputs (Hz)
    
    // ========================================
    // NEUROMODULATION (for Phase 4)
    // ========================================
    float dopamine_level;               // Local dopamine concentration
    float serotonin_level;              // Local serotonin concentration
    float acetylcholine_level;          // Local acetylcholine concentration
    float noradrenaline_level;          // Local noradrenaline concentration
    
    // Neuromodulator effects on channels
    float neuromod_ampa_scale;          // AMPA scaling factor
    float neuromod_nmda_scale;          // NMDA scaling factor
    float neuromod_gaba_scale;          // GABA scaling factor
    float neuromod_excitability;        // Overall excitability modulation
    
    // ========================================
    // DEVELOPMENT AND PLASTICITY STATE
    // ========================================
    int developmental_stage;            // Developmental stage (0=immature, 1=mature)
    float plasticity_threshold;         // Threshold for synaptic modifications
    float homeostatic_target;           // Target activity level for homeostasis
    float metaplasticity_state;         // Metaplasticity state variable
    
    // ========================================
    // COMPUTATIONAL EFFICIENCY FIELDS
    // ========================================
    float last_update_time;             // Time of last update
    bool needs_update;                  // Whether neuron needs updating
    int update_priority;                // Update priority (0=low, 1=high)
    
    // ========================================
    // DEBUGGING AND MONITORING
    // ========================================#ifndef GPU_NEURAL_STRUCTURES_H

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

    float max_voltage_reached;          // Maximum voltage in last timestep
    float total_current_injected;       // Total current from all sources
    int spike_count;                    // Total number of spikes
    float energy_consumption;           // Metabolic energy consumption
};

/**
 * Enhanced GPU synapse structure with receptor-specific targeting
 * Extended from Phase 1 to support new ion channel dynamics
 */
struct GPUSynapse {
    // ========================================
    // BASIC SYNAPSE PROPERTIES
    // ========================================
    int pre_neuron;                     // Legacy: ID of presynaptic neuron
    int post_neuron;                    // Legacy: ID of postsynaptic neuron
    int pre_neuron_idx;                 // Index of presynaptic neuron
    int post_neuron_idx;                // Index of postsynaptic neuron
    
    // ========================================
    // SYNAPTIC STRENGTH AND DYNAMICS
    // ========================================
    float weight;                       // Base synaptic weight
    float effective_weight;             // Modulated effective weight
    float max_weight;                   // Maximum allowed weight
    float min_weight;                   // Minimum allowed weight
    
    // ========================================
    // TEMPORAL DYNAMICS
    // ========================================
    float delay;                        // Synaptic delay (ms)
    float last_active;                  // Time of last activation
    float last_pre_spike_time;          // Time of last presynaptic spike
    float last_post_spike_time;         // Time of last postsynaptic spike
    
    // ========================================
    // TARGETING AND RECEPTOR SPECIFICITY (NEW)
    // ========================================
    int post_compartment;               // Target compartment index
    int receptor_index;                 // Target receptor type (AMPA/NMDA/GABA_A/GABA_B)
    float receptor_weight_fraction;     // Fraction of weight for this receptor
    
    // ========================================
    // SYNAPSE TYPE AND STATE
    // ========================================
    int type;                          // Synapse type (excitatory/inhibitory)
    int active;                        // Whether synapse is active
    bool is_plastic;                   // Whether synapse can change strength
    
    // ========================================
    // ACTIVITY METRICS
    // ========================================
    float activity_metric;             // Recent activity measure
    float release_probability;         // Probability of neurotransmitter release
    float last_potentiation;           // Time of last potentiation event
    float last_depression;             // Time of last depression event
    
    // ========================================
    // ENHANCED ELIGIBILITY TRACES (for Phase 3)
    // ========================================
    float eligibility_trace;           // Legacy eligibility trace
    float fast_trace;                  // Fast eligibility trace (10-100ms)
    float medium_trace;                // Medium eligibility trace (1-10s)
    float slow_trace;                  // Slow eligibility trace (minutes)
    float tag_strength;                // Synaptic tag for late-phase plasticity
    
    // ========================================
    // PLASTICITY PARAMETERS
    // ========================================
    float plasticity_rate;             // Learning rate for this synapse
    float meta_weight;                 // Metaplastic weight (controls plasticity)
    float recent_activity;             // Sliding window activity measure
    float calcium_trace;               // Calcium-dependent trace
    
    // ========================================
    // NEUROMODULATION SENSITIVITY
    // ========================================
    float dopamine_sensitivity;        // Sensitivity to dopamine modulation
    float plasticity_modulation;       // Current plasticity modulation level
    
    // ========================================
    // VESICLE DYNAMICS (for realistic transmission)
    // ========================================
    int vesicle_pool_size;             // Number of available vesicles
    int vesicles_ready;                // Number of ready-to-release vesicles
    float vesicle_recovery_rate;       // Rate of vesicle replenishment
    
    // ========================================
    // COMPUTATIONAL OPTIMIZATION
    // ========================================
    bool needs_plasticity_update;      // Whether plasticity needs updating
    float last_plasticity_update;      // Time of last plasticity calculation
    int plasticity_update_interval;    // Update interval for plasticity
};

/**
 * GPU spike event structure (unchanged from Phase 1)
 */
struct GPUSpikeEvent {
    int neuron_idx;                     // Index of spiking neuron
    float time;                         // Spike time (ms)
    float amplitude;                    // Spike amplitude (mV)
    int compartment_idx;                // Compartment that generated spike
};

/**
 * Ion channel parameter structure for initialization
 */
struct IonChannelParams {
    // AMPA parameters
    float ampa_g_max, ampa_tau_rise, ampa_tau_decay, ampa_reversal;
    
    // NMDA parameters  
    float nmda_g_max, nmda_tau_rise, nmda_tau_decay, nmda_reversal;
    float nmda_mg_conc, nmda_ca_fraction;
    
    // GABA-A parameters
    float gaba_a_g_max, gaba_a_tau_rise, gaba_a_tau_decay, gaba_a_reversal;
    
    // GABA-B parameters
    float gaba_b_g_max, gaba_b_tau_rise, gaba_b_tau_decay, gaba_b_tau_k, gaba_b_reversal;
    
    // Voltage-gated calcium channel parameters
    float ca_g_max, ca_reversal, ca_v_half, ca_k, ca_tau_act;
    
    // KCa channel parameters
    float kca_g_max, kca_reversal, kca_ca_half, kca_hill_coef, kca_tau_act;
    
    // HCN channel parameters
    float hcn_g_max, hcn_reversal, hcn_v_half, hcn_k;
    float hcn_tau_min, hcn_tau_max, hcn_v_tau, hcn_k_tau;
    
    // Calcium dynamics parameters
    float ca_resting, ca_buffer_capacity, ca_buffer_kd;
    float ca_extrusion_rate, ca_diffusion_rate, ca_volume_factor;
};

#endif // GPU_NEURAL_STRUCTURES_H