#ifndef LEARNING_RULE_CONSTANTS_H
#define LEARNING_RULE_CONSTANTS_H

// ========================================
// COMPARTMENT AND RECEPTOR TYPE DEFINITIONS
// ========================================
#define COMPARTMENT_SOMA 0
#define COMPARTMENT_BASAL 1
#define COMPARTMENT_APICAL 2
#define COMPARTMENT_AXON 3

#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3

#define NEURON_EXCITATORY 0
#define NEURON_INHIBITORY 1
#define NEURON_REWARD_PREDICTION 2

// ========================================
// ENHANCED STDP PARAMETERS
// ========================================
// Base STDP time constants (ms)
#define STDP_TAU_PLUS_BASE 20.0f
#define STDP_TAU_MINUS_BASE 20.0f

// Compartment-specific STDP modulation
#define STDP_BASAL_AMP_PLUS 0.005f      // Standard potentiation
#define STDP_BASAL_AMP_MINUS 0.0025f    // Standard depression
#define STDP_APICAL_AMP_PLUS 0.008f     // Enhanced apical potentiation
#define STDP_APICAL_AMP_MINUS 0.002f    // Reduced apical depression
#define STDP_SOMATIC_AMP_PLUS 0.003f    // Conservative somatic learning
#define STDP_SOMATIC_AMP_MINUS 0.003f   // Balanced somatic plasticity

// Receptor-specific STDP parameters
#define STDP_AMPA_MULTIPLIER 1.0f       // Standard AMPA plasticity
#define STDP_NMDA_MULTIPLIER 1.5f       // Enhanced NMDA plasticity
#define STDP_GABA_A_MULTIPLIER -0.3f    // Inhibitory plasticity
#define STDP_GABA_B_MULTIPLIER -0.2f    // Slower inhibitory plasticity

// Timing window parameters
#define STDP_MAX_DT 100.0f              // Maximum timing difference (ms)
#define STDP_MIN_DT 1.0f                // Minimum timing difference (ms)

// ========================================
// MULTI-TIMESCALE ELIGIBILITY TRACES
// ========================================
#define FAST_TRACE_TAU 50.0f            // Fast trace: immediate learning (ms)
#define MEDIUM_TRACE_TAU 5000.0f        // Medium trace: short-term memory (ms)
#define SLOW_TRACE_TAU 100000.0f        // Slow trace: long-term memory (ms)
#define CALCIUM_TRACE_TAU 200.0f        // Calcium dynamics (ms)

// Trace transfer rates
#define FAST_TO_MEDIUM_RATE 0.1f        // Fast → Medium transfer coefficient
#define MEDIUM_TO_SLOW_RATE 0.05f       // Medium → Slow transfer coefficient
#define TRACE_SATURATION_FACTOR 0.9f    // Prevents unbounded growth

// Maximum trace values
#define MAX_FAST_TRACE 2.0f
#define MAX_MEDIUM_TRACE 5.0f
#define MAX_SLOW_TRACE 10.0f
#define MAX_CALCIUM_TRACE 1.0f

// ========================================
// SYNAPTIC TAGGING AND CAPTURE
// ========================================
#define TAG_TAU 30000.0f                // Tag decay time constant (ms)
#define TAG_THRESHOLD 0.5f              // Threshold for tag creation
#define TAG_CREATION_RATE 0.1f          // Rate of tag formation
#define MAX_TAG_STRENGTH 1.0f           // Maximum tag strength
#define PROTEIN_SYNTHESIS_THRESHOLD 0.7f // Threshold for late-phase plasticity
#define LATE_PHASE_FACTOR 0.3f          // Late-phase plasticity magnitude

// ========================================
// REWARD MODULATION AND DOPAMINE
// ========================================
#define BASELINE_DOPAMINE 0.0f          // Baseline dopamine level
#define DOPAMINE_TAU 1000.0f            // Dopamine decay time constant (ms)
#define REWARD_PREDICTION_TAU 500.0f    // Prediction error time constant (ms)

// Trace modulation factors
#define FAST_TRACE_DOPAMINE_SENS 1.0f   // Fast trace dopamine sensitivity
#define MEDIUM_TRACE_DOPAMINE_SENS 0.5f // Medium trace dopamine sensitivity
#define SLOW_TRACE_DOPAMINE_SENS 0.1f   // Slow trace dopamine sensitivity

// Prediction error parameters
#define PREDICTION_ERROR_SCALE 2.0f     // Amplification of prediction errors
#define REWARD_DISCOUNT_FACTOR 0.95f    // Temporal discount for rewards
#define MIN_REWARD_THRESHOLD 0.01f      // Minimum reward for modulation

// ========================================
// HEBBIAN LEARNING MECHANISMS
// ========================================
#define HEBBIAN_LEARNING_RATE 0.0001f   // Base Hebbian learning rate
#define COVARIANCE_WINDOW 100.0f        // Time window for covariance (ms)
#define CORRELATION_THRESHOLD 0.1f      // Minimum correlation for Hebbian update
#define HEBBIAN_SATURATION 0.95f        // Prevents runaway potentiation

// Activity-dependent modulation
#define ACTIVITY_SCALING_FACTOR 0.1f    // Scales Hebbian updates by activity
#define MIN_ACTIVITY_THRESHOLD 0.01f    // Minimum activity for Hebbian learning

// ========================================
// HOMEOSTATIC MECHANISMS
// ========================================
#define TARGET_FIRING_RATE 10.0f        // Target firing rate (Hz)
#define FIRING_RATE_TAU 10000.0f        // Firing rate integration time (ms)
#define SYNAPTIC_SCALING_RATE 0.001f    // Rate of synaptic scaling
#define SYNAPTIC_SCALING_TAU 60000.0f   // Time constant for scaling (ms)

// Weight bounds and normalization
#define MIN_WEIGHT -2.0f                // Minimum synaptic weight
#define MAX_WEIGHT 2.0f                 // Maximum synaptic weight
#define MAX_TOTAL_IN_WEIGHT 10.0f       // Maximum total incoming weight
#define MAX_TOTAL_OUT_WEIGHT 20.0f      // Maximum total outgoing weight
#define WEIGHT_NORMALIZATION_RATE 0.01f // Rate of weight normalization

// Activity regulation
#define TARGET_ACTIVITY_LEVEL 0.1f      // Target activity level
#define ACTIVITY_TAU 1000.0f            // Activity integration time (ms)
#define INTRINSIC_EXCITABILITY_RATE 0.0001f // Rate of excitability changes

// ========================================
// METAPLASTICITY PARAMETERS
// ========================================
#define META_PLASTICITY_TAU 300000.0f   // Metaplasticity time constant (ms)
#define META_THRESHOLD_HIGH 0.8f        // High activity threshold
#define META_THRESHOLD_LOW 0.2f         // Low activity threshold
#define META_MODULATION_STRENGTH 0.5f   // Strength of metaplastic modulation
#define CALCIUM_THRESHOLD_LTD 0.2f      // Calcium threshold for LTD
#define CALCIUM_THRESHOLD_LTP 0.6f      // Calcium threshold for LTP

// ========================================
// COMPUTATIONAL PARAMETERS
// ========================================
#define LEARNING_RATE_DECAY 0.999f      // Gradual learning rate decay
#define NOISE_AMPLITUDE 0.001f          // Noise in plasticity updates
#define UPDATE_FREQUENCY 1.0f           // Plasticity update frequency (ms)
#define BATCH_UPDATE_SIZE 1000          // Number of synapses per batch

// Memory management
#define MAX_SPIKE_HISTORY 1000          // Maximum stored spike events
#define CLEANUP_INTERVAL 10000.0f       // Memory cleanup interval (ms)

// Numerical stability
#define EPSILON 1e-8f                   // Small value for numerical stability
#define MAX_WEIGHT_CHANGE 0.1f          // Maximum weight change per update

#endif // LEARNING_RULE_CONSTANTS_H