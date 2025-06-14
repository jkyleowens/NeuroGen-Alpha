#ifndef LEARNING_RULE_CONSTANTS_H
#define LEARNING_RULE_CONSTANTS_H

// ========================================
// NEURON AND COMPARTMENT TYPES
// ========================================
#define COMPARTMENT_SOMA 0
#define COMPARTMENT_BASAL 1
#define COMPARTMENT_APICAL 2
#define COMPARTMENT_AXON 3

#define NEURON_EXCITATORY 0
#define NEURON_INHIBITORY 1
#define NEURON_REWARD_PREDICTION 2

// ========================================
// SYNAPTIC WEIGHT & PLASTICITY CONSTRAINTS
// ========================================
#define MIN_WEIGHT -2.0f
#define MAX_WEIGHT 2.0f
#define MAX_WEIGHT_CHANGE 0.1f

// --- Synaptic Scaling ---
#define TARGET_FIRING_RATE 5.0f // Target firing rate in Hz for each neuron
#define FIRING_RATE_TAU 2000.0f // Time constant for averaging firing rate (ms)
#define SYNAPTIC_SCALING_RATE 0.002f // Rate of synaptic scaling adjustment

// --- Weight Normalization & E/I Balance ---
#define MAX_TOTAL_IN_WEIGHT 100.0f
#define MAX_TOTAL_OUT_WEIGHT 150.0f
#define MIN_WEIGHT -2.0f
#define MAX_WEIGHT 2.0f
#define E_I_RATIO_TARGET 4.0f // Target ratio of total excitatory to inhibitory input
#define EPSILON 1e-6f

// --- Intrinsic Plasticity ---
#define TARGET_ACTIVITY_LEVEL -60.0f // Target average membrane potential (mV)
#define ACTIVITY_TAU 1000.0f // Time constant for tracking average activity
#define INTRINSIC_EXCITABILITY_RATE 0.001f

// ========================================
// Spike-Timing-Dependent Plasticity (STDP)
// ========================================
// Time constants for STDP timing windows (ms)
#define STDP_TAU_PLUS 20.0f
#define STDP_TAU_MINUS 20.0f

// Learning rates (amplitudes) for excitatory synapses
#define STDP_A_PLUS_EXC 0.005f
#define STDP_A_MINUS_EXC 0.0025f

// Learning rates for inhibitory synapses (iSTDP)
#define STDP_A_PLUS_INH -0.001f  // Potentiation of inhibition
#define STDP_A_MINUS_INH 0.002f   // Depression of inhibition

// ========================================
// ELIGIBILITY TRACES & SYNAPTIC TAGGING
// ========================================
// Time constants (tau) in milliseconds
#define FAST_TRACE_TAU 50.0f
#define MEDIUM_TRACE_TAU 5000.0f
#define SLOW_TRACE_TAU 100000.0f
#define TAG_TAU 30000.0f

// Maximum trace values to prevent runaway feedback
#define MAX_FAST_TRACE 2.0f
#define MAX_MEDIUM_TRACE 5.0f
#define MAX_SLOW_TRACE 10.0f
#define MAX_TAG_STRENGTH 1.0f

// Parameters for synaptic tag creation and consolidation
#define TAG_THRESHOLD 0.5f
#define TAG_CREATION_RATE 0.1f
#define FAST_TO_MEDIUM_RATE 0.1f
#define MEDIUM_TO_SLOW_RATE 0.05f

// ========================================
// REWARD MODULATION & DOPAMINE SYSTEM
// ========================================
#define BASELINE_DOPAMINE 0.0f
#define DOPAMINE_TAU 1000.0f
#define REWARD_PREDICTION_TAU 500.0f
#define PREDICTION_ERROR_SCALE 2.0f
#define MIN_REWARD_THRESHOLD 0.01f

// Sensitivity of different traces to dopamine modulation
#define FAST_TRACE_DOPAMINE_SENS 1.0f
#define MEDIUM_TRACE_DOPAMINE_SENS 0.5f
#define SLOW_TRACE_DOPAMINE_SENS 0.1f

// Protein synthesis and late-phase plasticity
#define PROTEIN_SYNTHESIS_THRESHOLD 0.7f
#define LATE_PHASE_FACTOR 0.3f

// ========================================
// HEBBIAN & CALCIUM-DEPENDENT PLASTICITY
// ========================================
#define HEBBIAN_LEARNING_RATE 0.0001f
#define CORRELATION_THRESHOLD 0.1f
#define ACTIVITY_SCALING_FACTOR 0.1f
#define CA_THRESHOLD_LTD 0.2f
#define CA_THRESHOLD_LTP 0.5f
#define CALCIUM_TRACE_TAU 200.0f
#define MAX_CALCIUM_TRACE 1.0f
#define MIN_ACTIVITY_THRESHOLD 0.01f // Minimum activity for a neuron to be considered for plasticity


// ========================================
// HOMEOSTATIC MECHANISMS
// ========================================
#define TARGET_FIRING_RATE 10.0f
#define FIRING_RATE_TAU 10000.0f
#define SYNAPTIC_SCALING_RATE 0.001f
#define TARGET_ACTIVITY_LEVEL 0.1f
#define ACTIVITY_TAU 1000.0f
#define INTRINSIC_EXCITABILITY_RATE 0.0001f
#define MAX_TOTAL_IN_WEIGHT 10.0f
#define MAX_TOTAL_OUT_WEIGHT 20.0f

// ========================================
// NEUROMODULATION CONSTANTS
// ========================================
#define ACETYLCHOLINE_EXCITABILITY_FACTOR 0.4f
#define SEROTONIN_INHIBITORY_FACTOR 0.4f
#define ACETYLCHOLINE_PLASTICITY_FACTOR 0.4f

// ========================================
// METAPLASTICITY
// ========================================
#define META_THRESHOLD_HIGH 0.8f
#define META_THRESHOLD_LOW 0.2f

// ========================================
// NUMERICAL STABILITY
// ========================================
#define EPSILON 1e-8f

#endif // LEARNING_RULE_CONSTANTS_H