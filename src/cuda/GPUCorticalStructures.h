#pragma once
#ifndef GPU_CORTICAL_STRUCTURES_H
#define GPU_CORTICAL_STRUCTURES_H

#include <cuda_runtime.h>

// Enhanced neuron types for cortical modeling
enum class NeuronType : unsigned char {
    PYRAMIDAL_L23 = 0,    // Layer 2/3 Pyramidal
    PYRAMIDAL_L5 = 1,     // Layer 5 Pyramidal  
    PYRAMIDAL_L6 = 2,     // Layer 6 Pyramidal
    STELLATE_L4 = 3,      // Layer 4 Stellate
    INTERNEURON_FS = 4,   // Fast-spiking interneuron
    INTERNEURON_RS = 5,   // Regular-spiking interneuron
    INTERNEURON_IS = 6    // Irregular-spiking interneuron
};

enum class CorticalLayer : unsigned char {
    LAYER_1 = 1,
    LAYER_2 = 2, 
    LAYER_3 = 3,
    LAYER_4 = 4,
    LAYER_5 = 5,
    LAYER_6 = 6
};

// Enhanced neuron state with cortical column specificity
struct GPUCorticalNeuron {
    // Core HH dynamics (preserved from original)
    float voltage;
    float m, h, n;                    // HH gating variables
    bool spiked;
    float last_spike_time;
    
    // Multi-compartment dynamics (enhanced)
    float soma_voltage;               // Somatic voltage
    float dendrite_voltage[4];        // Up to 4 dendritic compartments
    float axon_voltage;               // Axonal voltage
    
    // Cortical-specific properties
    NeuronType cell_type;
    CorticalLayer layer;
    unsigned short column_id;         // Which column this neuron belongs to
    unsigned short minicolumn_id;     // Which minicolumn within the column
    unsigned char layer_position;     // Position within layer (0-255)
    
    // Enhanced ionic currents
    float I_Na, I_K, I_L;            // Standard HH currents
    float I_Ca, I_KCa, I_M;          // Additional currents for pyramidal cells
    float I_h;                       // Hyperpolarization-activated current
    
    // Synaptic integration
    float I_AMPA, I_NMDA;            // Excitatory currents
    float I_GABA_A, I_GABA_B;        // Inhibitory currents
    
    // Adaptation and plasticity
    float adaptation_current;         // Spike-frequency adaptation
    float Ca_concentration;           // Intracellular calcium
    float plasticity_threshold;       // Dynamic plasticity threshold
    
    // Burst firing dynamics (for pyramidal cells)
    bool in_burst;
    unsigned char burst_count;
    float burst_threshold;
    
    // Activity history for homeostasis
    float activity_trace;             // Exponential trace of recent activity
    float firing_rate_avg;            // Running average firing rate
    
    // Connectivity metadata
    unsigned short num_excitatory_inputs;
    unsigned short num_inhibitory_inputs;
    
    // Timing and synchronization
    float phase_preference;           // Preferred oscillation phase
    float last_burst_time;
};

// Enhanced synapse structure for cortical connectivity
struct GPUCorticalSynapse {
    unsigned int pre_neuron_idx;
    unsigned int post_neuron_idx;
    
    // Basic synaptic properties  
    float weight;
    float delay;                      // Propagation delay (1-20ms)
    float max_weight;                 // Upper bound for plasticity
    float min_weight;                 // Lower bound for plasticity
    
    // Synaptic type and dynamics
    enum SynapseType : unsigned char {
        AMPA = 0, NMDA = 1, GABA_A = 2, GABA_B = 3
    } synapse_type;
    
    // STDP parameters (enhanced)
    float A_plus, A_minus;            // Potentiation/depression amplitudes
    float tau_plus, tau_minus;        // Time constants
    float last_pre_spike_time;
    float last_post_spike_time;
    
    // Short-term plasticity
    float u_0;                        // Initial release probability
    float tau_f, tau_d;               // Facilitation/depression time constants
    float u_current;                  // Current release probability  
    float R_current;                  // Current available resources
    
    // Neuromodulation
    float dopamine_sensitivity;       // Sensitivity to dopaminergic modulation
    float acetylcholine_sensitivity;  // Sensitivity to cholinergic modulation
    
    // Connection specificity
    CorticalLayer pre_layer;
    CorticalLayer post_layer;
    unsigned short pre_column_id;
    unsigned short post_column_id;
    
    // Activity tracking
    float activity_metric;
    float correlation_trace;          // Pre-post correlation history
};

// Cortical column structure
struct GPUCorticalColumn {
    unsigned short column_id;
    
    // Spatial organization
    float center_x, center_y;         // Column center coordinates
    float radius;                     // Column radius (typically 150μm)
    
    // Neuron organization
    unsigned short neuron_start_idx;  // First neuron index in this column
    unsigned short neuron_count;      // Total neurons in this column
    
    // Layer-specific neuron counts
    unsigned char layer_neuron_counts[6]; // Neurons per layer
    unsigned short layer_start_indices[6]; // Starting indices per layer
    
    // Minicolumn organization
    unsigned char minicolumn_count;   // Number of minicolumns
    unsigned short minicolumn_size;   // Neurons per minicolumn
    
    // Column-level dynamics
    float column_activity;            // Overall activity level
    float dominant_frequency;         // Current dominant oscillation
    float phase_coherence;            // Phase synchronization measure
    
    // Connectivity statistics
    unsigned int internal_synapses;   // Intra-column connections
    unsigned int external_synapses;   // Inter-column connections
    
    // Learning and adaptation
    float column_learning_rate;       // Adaptive learning rate
    float homeostatic_target;         // Target activity level
    float plasticity_threshold;       // Column-wide plasticity modulation
    
    // Neuromodulation levels
    float dopamine_level;
    float acetylcholine_level;
    float norepinephrine_level;
};

// Spike event with cortical context
struct GPUCorticalSpikeEvent {
    unsigned int neuron_index;
    float spike_time;
    NeuronType neuron_type;
    CorticalLayer layer;
    unsigned short column_id;
    unsigned short minicolumn_id;
    bool is_burst_spike;              // Part of a burst
    float spike_amplitude;            // Variable spike heights
};

// Oscillation tracking for network dynamics
struct GPUOscillationState {
    // Frequency bands
    float gamma_power;                // 30-100 Hz
    float beta_power;                 // 13-30 Hz  
    float alpha_power;                // 8-13 Hz
    float theta_power;                // 4-8 Hz
    
    // Phase information
    float gamma_phase;
    float beta_phase;
    float alpha_phase;
    float theta_phase;
    
    // Cross-frequency coupling
    float theta_gamma_coupling;
    float alpha_beta_coupling;
    
    // Synchronization measures
    float local_synchrony;            // Within-column synchronization
    float global_synchrony;           // Between-column synchronization
};

// Network-level cortical state
struct GPUCorticalNetworkState {
    unsigned short num_columns;
    unsigned int total_neurons;
    unsigned int total_synapses;
    
    // Global neuromodulation
    float global_dopamine;
    float global_acetylcholine;
    float global_norepinephrine;
    
    // Network oscillations
    GPUOscillationState oscillation_state;
    
    // Learning state
    float global_learning_rate;
    bool stdp_enabled;
    bool homeostasis_enabled;
    
    // Performance metrics
    float average_firing_rate;
    float synchronization_index;
    float information_capacity;
};

#endif // GPU_CORTICAL_STRUCTURES_H