#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include <cuda_runtime.h>
#include <NeuroGen/IonChannelConstants.h>

// Maximum number of compartments per neuron
#define MAX_COMPARTMENTS 8

/**
 * @struct GPUNeuronState
 * @brief GPU-optimized structure for individual neurons with enhanced ion channel dynamics.
 *
 * Extended from Phase 1 to include comprehensive ion channel states, this structure
 * holds all state variables for a single neuron required for simulation on the GPU.
 */
struct GPUNeuronState {
    // ========================================
    // BASIC NEURON PROPERTIES
    // ========================================
    int neuron_id;                          // Unique identifier for the neuron
    int active;                             // Status of the neuron (1 for active, 0 for inactive)
    int type;                               // Neuron type, e.g., excitatory or inhibitory

    // ========================================
    // COMPARTMENT STRUCTURE
    // ========================================
    int compartment_count;                          // Number of compartments in this neuron
    int compartment_types[MAX_COMPARTMENTS];        // The type of each compartment
    int parent_compartment[MAX_COMPARTMENTS];       // Index of the parent compartment (-1 for soma)
    float coupling_conductance[MAX_COMPARTMENTS];   // Conductance for coupling to the parent

    // ========================================
    // VOLTAGE AND HODGKIN-HUXLEY DYNAMICS
    // ========================================
    float voltage;                                  // Membrane potential of the soma (mV)
    float voltages[MAX_COMPARTMENTS];               // Voltage for each compartment (mV)
    
    // Hodgkin-Huxley state variables
    float m, h, n;                                  // Soma HH variables
    float m_comp[MAX_COMPARTMENTS];                 // Sodium activation (m) for each compartment
    float h_comp[MAX_COMPARTMENTS];                 // Sodium inactivation (h) for each compartment
    float n_comp[MAX_COMPARTMENTS];                 // Potassium activation (n) for each compartment
    
    // ========================================
    // ION CHANNEL STATES (Phase 2 Enhancement)
    // ========================================
    struct {
        // Synaptic receptor states
        float ampa_g[MAX_COMPARTMENTS];             // AMPA conductance
        float ampa_state[MAX_COMPARTMENTS];         // AMPA state variable
        float nmda_g[MAX_COMPARTMENTS];             // NMDA conductance
        float nmda_state[MAX_COMPARTMENTS];         // NMDA state variable
        float gaba_a_g[MAX_COMPARTMENTS];           // GABA-A conductance
        float gaba_a_state[MAX_COMPARTMENTS];       // GABA-A state variable
        float gaba_b_g[MAX_COMPARTMENTS];           // GABA-B conductance
        float gaba_b_state[MAX_COMPARTMENTS];       // GABA-B state variable
        float gaba_b_g_protein[MAX_COMPARTMENTS];   // GABA-B G-protein activation
        
        // Voltage-gated channel states
        float ca_m[MAX_COMPARTMENTS];               // Calcium channel activation
        float kca_m[MAX_COMPARTMENTS];              // KCa channel activation (calcium-dependent potassium)
        float hcn_h[MAX_COMPARTMENTS];              // HCN channel activation (hyperpolarization-activated cyclic nucleotide-gated)
    } channels;
    
    // ========================================
    // CALCIUM DYNAMICS (Phase 2 Enhancement)
    // ========================================
    float ca_conc[MAX_COMPARTMENTS];                // Intracellular calcium concentration (mM)
    float ca_buffer[MAX_COMPARTMENTS];              // Concentration of calcium-bound buffer
    float ca_pump_rate[MAX_COMPARTMENTS];           // Rate of calcium extrusion (1/ms)
    
    // ========================================
    // LEGACY RECEPTOR CONDUCTANCES
    // ========================================
    // Maintained for backward compatibility. Will be phased out.
    float receptor_conductances[MAX_COMPARTMENTS][NUM_RECEPTOR_TYPES];
    
    // ========================================
    // SPIKE GENERATION AND TIMING
    // ========================================
    bool spiked;                                    // Flag indicating a spike in the current timestep
    float last_spike_time;                          // Timestamp of the last spike (ms)
    float spike_threshold;                          // Base threshold for spiking (mV)
    float spike_threshold_modulated;                // Spike threshold after neuromodulation (mV)
    float refractory_period;                        // Duration of the refractory period (ms)
    float time_since_spike;                         // Time elapsed since the last spike (ms)
    
    // ========================================
    // ACTIVITY AND PLASTICITY METRICS
    // ========================================
    float activity_level;                           // Recent activity level (normalized 0-1)
    float avg_firing_rate;                          // Average firing rate (Hz)
    float membrane_resistance;                      // Input resistance (MΩ)
    float membrane_capacitance;                     // Membrane capacitance (pF)


    // ========================================
    // HOMEOSTATIC REGULATION (NEWLY ADDED)
    // ========================================
    float average_activity;             // Longer-term average activity level
    float homeostatic_scaling_factor;   // Synaptic scaling factor for this neuron
    float adaptation_current;           // Current level of spike-frequency adaptation
    float leak_conductance;             // Base leak conductance for the membrane
    float homeostatic_time_constant;    // Time constant for homeostatic adjustments
    float average_firing_rate;
    float threshold;
    

    
    // ========================================
    // DENDRITIC PROCESSING
    // ========================================
    bool dendritic_spike[MAX_COMPARTMENTS];         // Flags for dendritic spikes
    float dendritic_spike_time[MAX_COMPARTMENTS];   // Timestamps of last dendritic spikes
    float dendritic_threshold[MAX_COMPARTMENTS];    // Thresholds for dendritic spikes
    
    // ========================================
    // SYNAPTIC INPUT TRACKING
    // ========================================
    float total_excitatory_input;                   // Sum of excitatory synaptic currents
    float total_inhibitory_input;                   // Sum of inhibitory synaptic currents
    float synaptic_input_rate;                      // Rate of incoming synaptic events (Hz)
    
    // ========================================
    // NEUROMODULATION (Phase 4)
    // ========================================
    float dopamine_level;                           // Local dopamine concentration
    float serotonin_level;                          // Local serotonin concentration
    float acetylcholine_level;                      // Local acetylcholine concentration
    float noradrenaline_level;                      // Local noradrenaline concentration
    
    // Neuromodulator effects
    float neuromod_ampa_scale;                      // Scaling factor for AMPA channels
    float neuromod_nmda_scale;                      // Scaling factor for NMDA channels
    float neuromod_gaba_scale;                      // Scaling factor for GABA channels
    float neuromod_excitability;                    // Overall modulation of neuron excitability
    float neuromod_adaptation;                      // Modulation of spike-frequency adaptation
    
    // ========================================
    // DEVELOPMENT AND PLASTICITY STATE
    // ========================================
    int developmental_stage;                        // Developmental stage (e.g., immature, mature)
    float plasticity_threshold;                     // Threshold for triggering synaptic modifications
    float homeostatic_target;                       // Target activity level for homeostatic regulation
    float metaplasticity_state;                     // State variable for metaplasticity
    
    // ========================================
    // COMPUTATIONAL EFFICIENCY FIELDS
    // ========================================
    float last_update_time;                         // Timestamp of the last state update
    bool needs_update;                              // Flag indicating if an update is needed
    int update_priority;                            // Priority for updates (0=low, 1=high)

    // ========================================
    // DIAGNOSTIC METRICS
    // ========================================
    float max_voltage_reached;                      // Maximum voltage in the last timestep
    float total_current_injected;                   // Total current from all sources
    int spike_count;                                // Total number of spikes since simulation start
    float energy_consumption;                       // Estimated metabolic energy consumption
};


/**
 * @struct GPUSynapse
 * @brief Enhanced GPU synapse structure with receptor-specific targeting.
 *
 * This structure contains all necessary parameters for a synapse, including its strength,
 * temporal dynamics, plasticity mechanisms, and sensitivity to neuromodulation.
 */
struct GPUSynapse {
    // ========================================
    // BASIC SYNAPSE PROPERTIES
    // ========================================
    int pre_neuron_idx;                 // Index of the presynaptic neuron
    int post_neuron_idx;                // Index of the postsynaptic neuron
    
    // ========================================
    // SYNAPTIC STRENGTH AND DYNAMICS
    // ========================================
    float weight;                       // Base synaptic weight
    float effective_weight;             // Weight after modulation
    float max_weight;                   // Maximum allowed weight
    float min_weight;                   // Minimum allowed weight
    
    // ========================================
    // TEMPORAL DYNAMICS
    // ========================================
    float delay;                        // Synaptic delay (ms)
    float last_active;                  // Time of last activation (ms)
    float last_pre_spike_time;          // Time of the last presynaptic spike (ms)
    float last_post_spike_time;         // Time of the last postsynaptic spike (ms)
    
    // ========================================
    // TARGETING AND RECEPTOR SPECIFICITY
    // ========================================
    int post_compartment;               // Target compartment index on the postsynaptic neuron
    int receptor_index;                 // Target receptor type (e.g., AMPA, NMDA)
    float receptor_weight_fraction;     // Fraction of the weight applied to this receptor
    
    // ========================================
    // SYNAPSE TYPE AND STATE
    // ========================================
    int type;                           // Synapse type (e.g., excitatory, inhibitory)
    int active;                         // Active status (1 for active, 0 for inactive)
    bool is_plastic;                    // Whether the synapse can undergo plasticity
    
    // ========================================
    // ACTIVITY METRICS
    // ========================================
    float activity_metric;              // A measure of recent activity
    float release_probability;          // Probability of neurotransmitter release
    float last_potentiation;            // Time of the last potentiation event (ms)
    float last_depression;              // Time of the last depression event (ms)
    
    // ========================================
    // ENHANCED ELIGIBILITY TRACES (Phase 3)
    // ========================================
    float eligibility_trace;            // Legacy eligibility trace for basic STDP
    float fast_trace;                   // Fast trace for rapid events (10-100ms)
    float medium_trace;                 // Medium-term trace for consolidation (1-10s)
    float slow_trace;                   // Slow trace for long-term structural changes (minutes)
    float tag_strength;                 // Synaptic tag for late-phase plasticity
    
    // ========================================
    // PLASTICITY PARAMETERS
    // ========================================
    float plasticity_rate;              // Learning rate for this synapse
    float meta_weight;                  // Metaplastic weight, controlling plasticity threshold
    float recent_activity;              // A sliding window measure of activity
    float calcium_trace;                // Trace of local calcium, driving plasticity
    
    // ========================================
    // NEUROMODULATION SENSITIVITY (Phase 4)
    // ========================================
    float dopamine_sensitivity;         // Sensitivity to dopamine (default: 1.0)
    float acetylcholine_sensitivity;    // Sensitivity to acetylcholine (default: 1.0)
    float serotonin_sensitivity;        // Sensitivity to serotonin (default: 1.0)
    float noradrenaline_sensitivity;    // Sensitivity to noradrenaline (default: 1.0)
    float plasticity_modulation;        // The final combined effect of neuromodulators on the learning rate
    float homeostatic_scaling_factor;   // Multiplicative scaling factor for stability

    // ========================================
    // VESICLE DYNAMICS
    // ========================================
    int vesicle_pool_size;              // Total number of neurotransmitter vesicles
    int vesicles_ready;                 // Number of vesicles ready for release
    float vesicle_recovery_rate;        // Rate of vesicle replenishment
    
    // ========================================
    // COMPUTATIONAL OPTIMIZATION
    // ========================================
    bool needs_plasticity_update;       // Flag for whether plasticity needs updating
    float last_plasticity_update;       // Time of the last plasticity calculation
    int plasticity_update_interval;     // Interval for plasticity updates (in timesteps)
};

/**
 * @struct GPUSpikeEvent
 * @brief GPU-optimized structure for spike events.
 *
 * Represents a single spike event to be processed on the GPU.
 */
struct GPUSpikeEvent {
    int neuron_idx;                     // Index of the neuron that spiked
    float time;                         // Time of the spike (ms)
    float amplitude;                    // Amplitude of the spike (mV)
    int compartment_idx;                // Index of the compartment that generated the spike
};

/**
 * @struct IonChannelParams
 * @brief Structure to hold initialization parameters for various ion channels.
 *
 * This allows for easy configuration and passing of channel parameters to GPU kernels.
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
    
    // KCa channel parameters (calcium-dependent potassium)
    float kca_g_max, kca_reversal, kca_ca_half, kca_hill_coef, kca_tau_act;
    
    // HCN channel parameters
    float hcn_g_max, hcn_reversal, hcn_v_half, hcn_k;
    float hcn_tau_min, hcn_tau_max, hcn_v_tau, hcn_k_tau;
    
    // Calcium dynamics parameters
    float ca_resting, ca_buffer_capacity, ca_buffer_kd;
    float ca_extrusion_rate, ca_diffusion_rate, ca_volume_factor;
};

#endif // GPU_NEURAL_STRUCTURES_H