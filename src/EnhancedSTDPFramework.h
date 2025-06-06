// EnhancedSTDPFramework.h
#ifndef ENHANCED_STDP_FRAMEWORK_H
#define ENHANCED_STDP_FRAMEWORK_H

#include "GPUNeuralStructures.h"
#include "IonChannelModels.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Multi-factor STDP framework implementing biologically realistic plasticity
 * Combines calcium-dependent thresholds, eligibility traces, and neuromodulation
 */

// ========================================
// PLASTICITY CONSTANTS AND PARAMETERS
// ========================================

// Calcium-dependent plasticity thresholds
#define CA_THRESHOLD_LTD_LOW    0.0002f   // 0.2 μM - below this = no change
#define CA_THRESHOLD_LTP_LOW    0.0005f   // 0.5 μM - LTD threshold
#define CA_THRESHOLD_LTP_HIGH   0.001f    // 1.0 μM - LTP threshold
#define CA_THRESHOLD_LTP_SAT    0.005f    // 5.0 μM - saturation threshold

// STDP timing windows
#define STDP_WINDOW_PRE_POST    20.0f     // ms - LTP window
#define STDP_WINDOW_POST_PRE    20.0f     // ms - LTD window
#define STDP_WINDOW_EXTENDED    100.0f    // ms - extended window for meta-plasticity

// Eligibility trace time constants
#define ELIGIBILITY_FAST_TAU    50.0f     // ms - fast synaptic trace
#define ELIGIBILITY_MEDIUM_TAU  1000.0f   // ms - medium-term trace
#define ELIGIBILITY_SLOW_TAU    60000.0f  // ms - slow structural trace
#define ELIGIBILITY_TAG_TAU     3600000.0f // ms - synaptic tag (1 hour)

// Learning rate parameters
#define STDP_LEARNING_RATE_BASE 0.001f    // Base learning rate
#define STDP_LEARNING_RATE_MAX  0.01f     // Maximum learning rate
#define METAPLASTICITY_FACTOR   2.0f      // Meta-plasticity scaling

// Homeostatic parameters
#define HOMEOSTATIC_TARGET_RATE 5.0f      // Hz - target firing rate
#define HOMEOSTATIC_TIME_WINDOW 10000.0f  // ms - averaging window
#define HOMEOSTATIC_SCALING_RATE 0.0001f  // Scaling adjustment rate

/**
 * Enhanced plasticity state structure for each synapse
 */
struct PlasticityState {
    // Multi-timescale eligibility traces
    float fast_eligibility;        // Fast synaptic trace (10-100ms)
    float medium_eligibility;      // Medium trace (1-10s)  
    float slow_eligibility;        // Slow structural trace (minutes)
    float synaptic_tag;            // Long-term synaptic tag (hours)
    
    // Calcium-dependent variables
    float calcium_integral;        // Integrated calcium signal
    float calcium_threshold;       // Dynamic plasticity threshold
    float ltp_magnitude;          // Magnitude of last LTP event
    float ltd_magnitude;          // Magnitude of last LTD event
    
    // Meta-plasticity state
    float metaplasticity_factor;   // Current meta-plasticity scaling
    float recent_activity;         // Recent synaptic activity level
    float potentiation_history;    // History of potentiation events
    float depression_history;      // History of depression events
    
    // Homeostatic variables
    float activity_integral;       // Integrated activity for homeostasis
    float scaling_factor;          // Homeostatic scaling factor
    float target_weight;           // Target weight for homeostasis
    
    // Timing variables
    float last_pre_spike;          // Time of last presynaptic spike
    float last_post_spike;         // Time of last postsynaptic spike
    float last_plasticity_event;   // Time of last plasticity change
    
    // Neuromodulation sensitivity
    float dopamine_sensitivity;    // Sensitivity to dopamine modulation
    float reward_prediction_error; // Current RPE signal
    float reward_eligibility;      // Reward-modulated eligibility
    
    // Structural plasticity markers
    float structural_stability;    // Resistance to structural changes
    float growth_signal;          // Signal for synapse strengthening
    float pruning_signal;         // Signal for synapse elimination
};

/**
 * STDP rule configuration structure
 */
struct STDPRuleConfig {
    // Timing-dependent parameters
    float pre_post_amp_positive;   // LTP amplitude for pre-before-post
    float pre_post_amp_negative;   // LTD amplitude for pre-before-post
    float post_pre_amp_positive;   // LTP amplitude for post-before-pre
    float post_pre_amp_negative;   // LTD amplitude for post-before-pre
    
    // Time constant parameters
    float tau_positive;           // Time constant for LTP
    float tau_negative;           // Time constant for LTD
    float tau_x;                  // Pre-synaptic trace time constant
    float tau_y;                  // Post-synaptic trace time constant
    
    // Calcium-dependent parameters
    float ca_amplitude_factor;    // Scaling by calcium levels
    float ca_threshold_factor;    // Threshold modulation by calcium
    float ca_cooperativity;       // Calcium cooperativity factor
    
    // Frequency-dependent parameters
    float frequency_factor;       // Frequency-dependent scaling
    float burst_factor;           // Burst-dependent enhancement
    float pattern_sensitivity;    // Sensitivity to spike patterns
    
    // Compartment-specific parameters
    float soma_scaling;          // Scaling for somatic synapses
    float basal_scaling;         // Scaling for basal dendritic synapses
    float apical_scaling;        // Scaling for apical dendritic synapses
    float spine_scaling;         // Scaling for spine synapses
    
    // Receptor-specific parameters
    float ampa_plasticity_factor; // AMPA receptor plasticity scaling
    float nmda_plasticity_factor; // NMDA receptor plasticity scaling
    float gaba_plasticity_factor; // GABA receptor plasticity scaling
    
    // Learning rate modulation
    float base_learning_rate;     // Base learning rate
    float max_learning_rate;      // Maximum learning rate
    float learning_rate_decay;    // Learning rate decay factor
};

/**
 * CUDA kernel for enhanced STDP computation
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    PlasticityState* plasticity_states,
    STDPRuleConfig* stdp_config,
    float* global_neuromodulators,
    float current_time,
    float dt,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[synapse_idx];
    PlasticityState& plasticity = plasticity_states[synapse_idx];
    
    // Skip inactive or non-plastic synapses
    if (synapse.active == 0 || !synapse.is_plastic) return;
    
    // Get pre- and post-synaptic neurons
    GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    int target_compartment = synapse.post_compartment;
    
    // ========================================
    // UPDATE ELIGIBILITY TRACES
    // ========================================
    
    // Decay all eligibility traces
    plasticity.fast_eligibility *= expf(-dt / ELIGIBILITY_FAST_TAU);
    plasticity.medium_eligibility *= expf(-dt / ELIGIBILITY_MEDIUM_TAU);
    plasticity.slow_eligibility *= expf(-dt / ELIGIBILITY_SLOW_TAU);
    plasticity.synaptic_tag *= expf(-dt / ELIGIBILITY_TAG_TAU);
    
    // Update traces based on spike activity
    bool pre_spike = (current_time - synapse.last_pre_spike_time) < dt;
    bool post_spike = false;
    
    // Check for postsynaptic spikes (soma and dendritic)
    if ((current_time - post_neuron.last_spike_time) < dt) {
        post_spike = true;
    } else if (target_compartment > 0 && 
               post_neuron.dendritic_spike[target_compartment] &&
               (current_time - post_neuron.dendritic_spike_time[target_compartment]) < dt) {
        post_spike = true;
    }
    
    // Update eligibility traces on spike events
    if (pre_spike) {
        plasticity.fast_eligibility += 1.0f;
        plasticity.medium_eligibility += 1.0f;
        plasticity.slow_eligibility += 1.0f;
        plasticity.last_pre_spike = current_time;
    }
    
    if (post_spike) {
        plasticity.last_post_spike = current_time;
    }
    
    // ========================================
    // CALCIUM-DEPENDENT PLASTICITY COMPUTATION
    // ========================================
    
    // Get local calcium concentration
    float local_calcium = post_neuron.ca_conc[target_compartment];
    
    // Update calcium integral with temporal dynamics
    float ca_decay_factor = expf(-dt / 200.0f);  // 200ms calcium integration window
    plasticity.calcium_integral = plasticity.calcium_integral * ca_decay_factor + 
                                 local_calcium * dt;
    
    // Determine plasticity direction based on calcium thresholds
    float plasticity_direction = 0.0f;
    float plasticity_magnitude = 0.0f;
    
    if (plasticity.calcium_integral > CA_THRESHOLD_LTP_HIGH) {
        // Strong LTP
        plasticity_direction = 1.0f;
        plasticity_magnitude = 1.0f + (plasticity.calcium_integral - CA_THRESHOLD_LTP_HIGH) / 
                               (CA_THRESHOLD_LTP_SAT - CA_THRESHOLD_LTP_HIGH);
        plasticity_magnitude = fminf(plasticity_magnitude, 3.0f);  // Cap at 3x
    } else if (plasticity.calcium_integral > CA_THRESHOLD_LTP_LOW) {
        // Weak LTP
        plasticity_direction = 1.0f;
        plasticity_magnitude = (plasticity.calcium_integral - CA_THRESHOLD_LTP_LOW) / 
                              (CA_THRESHOLD_LTP_HIGH - CA_THRESHOLD_LTP_LOW);
    } else if (plasticity.calcium_integral > CA_THRESHOLD_LTD_LOW) {
        // LTD
        plasticity_direction = -1.0f;
        plasticity_magnitude = (plasticity.calcium_integral - CA_THRESHOLD_LTD_LOW) / 
                              (CA_THRESHOLD_LTP_LOW - CA_THRESHOLD_LTD_LOW);
    }
    // Below LTD threshold = no plasticity
    
    // ========================================
    // SPIKE-TIMING DEPENDENT PLASTICITY
    // ========================================
    
    float stdp_magnitude = 0.0f;
    
    if (pre_spike && post_spike) {
        // Coincident spikes - strong LTP
        stdp_magnitude = stdp_config->pre_post_amp_positive * 2.0f;
    } else if (pre_spike) {
        // Pre-synaptic spike - check for recent post-synaptic activity
        float dt_post_pre = current_time - plasticity.last_post_spike;
        if (dt_post_pre >= 0 && dt_post_pre <= STDP_WINDOW_PRE_POST) {
            // Pre-before-post: LTP
            float time_factor = expf(-dt_post_pre / stdp_config->tau_positive);
            stdp_magnitude = stdp_config->pre_post_amp_positive * time_factor;
        } else if (dt_post_pre < 0 && fabs(dt_post_pre) <= STDP_WINDOW_POST_PRE) {
            // Post-before-pre: LTD
            float time_factor = expf(-fabs(dt_post_pre) / stdp_config->tau_negative);
            stdp_magnitude = -stdp_config->post_pre_amp_negative * time_factor;
        }
    }
    
    // ========================================
    // META-PLASTICITY MODULATION
    // ========================================
    
    // Update meta-plasticity based on recent activity
    float recent_activity_decay = expf(-dt / 10000.0f);  // 10s time constant
    plasticity.recent_activity = plasticity.recent_activity * recent_activity_decay + 
                                 (pre_spike ? 1.0f : 0.0f) * dt;
    
    // Meta-plasticity factor based on recent activity
    float meta_factor = 1.0f;
    if (plasticity.recent_activity > 0.1f) {
        // High activity - reduce plasticity (homeostatic)
        meta_factor = 1.0f / (1.0f + METAPLASTICITY_FACTOR * plasticity.recent_activity);
    } else {
        // Low activity - increase plasticity
        meta_factor = 1.0f + METAPLASTICITY_FACTOR * (0.1f - plasticity.recent_activity);
    }
    plasticity.metaplasticity_factor = meta_factor;
    
    // ========================================
    // NEUROMODULATION EFFECTS
    // ========================================
    
    // Get global neuromodulator levels
    float dopamine_level = global_neuromodulators[0];
    float serotonin_level = global_neuromodulators[1];
    float acetylcholine_level = global_neuromodulators[2];
    float noradrenaline_level = global_neuromodulators[3];
    
    // Dopamine modulation (reward prediction error)
    float dopamine_modulation = 1.0f + plasticity.dopamine_sensitivity * 
                               (dopamine_level - 0.5f);  // Baseline at 0.5
    
    // Acetylcholine enhances plasticity (attention/learning)
    float acetylcholine_modulation = 1.0f + 0.5f * acetylcholine_level;
    
    // Combined neuromodulation factor
    float neuromod_factor = dopamine_modulation * acetylcholine_modulation;
    neuromod_factor = fmaxf(0.1f, fminf(5.0f, neuromod_factor));  // Clamp to reasonable range
    
    // ========================================
    // COMBINED PLASTICITY COMPUTATION
    // ========================================
    
    // Combine calcium-dependent and STDP components
    float total_plasticity = 0.0f;
    
    if (plasticity_magnitude > 0.0f) {
        // Weight calcium and timing components
        float calcium_component = plasticity_direction * plasticity_magnitude * 
                                 stdp_config->ca_amplitude_factor;
        float stdp_component = stdp_magnitude;
        
        // Combine with eligibility traces
        float eligibility_factor = plasticity.fast_eligibility * 0.5f + 
                                  plasticity.medium_eligibility * 0.3f + 
                                  plasticity.slow_eligibility * 0.2f;
        eligibility_factor = fmaxf(0.1f, fminf(2.0f, eligibility_factor));
        
        total_plasticity = (calcium_component + stdp_component) * 
                          eligibility_factor * meta_factor * neuromod_factor;
    }
    
    // ========================================
    // APPLY WEIGHT CHANGES
    // ========================================
    
    if (fabs(total_plasticity) > 1e-6f) {  // Only apply significant changes
        // Scale by learning rate and receptor-specific factors
        float learning_rate = stdp_config->base_learning_rate;
        
        // Receptor-specific scaling
        switch (synapse.receptor_index) {
            case RECEPTOR_AMPA:
                learning_rate *= stdp_config->ampa_plasticity_factor;
                break;
            case RECEPTOR_NMDA:
                learning_rate *= stdp_config->nmda_plasticity_factor;
                break;
            case RECEPTOR_GABA_A:
            case RECEPTOR_GABA_B:
                learning_rate *= stdp_config->gaba_plasticity_factor;
                break;
        }
        
        // Compartment-specific scaling
        switch (post_neuron.compartment_types[target_compartment]) {
            case COMPARTMENT_SOMA:
                learning_rate *= stdp_config->soma_scaling;
                break;
            case COMPARTMENT_BASAL:
                learning_rate *= stdp_config->basal_scaling;
                break;
            case COMPARTMENT_APICAL:
                learning_rate *= stdp_config->apical_scaling;
                break;
            case COMPARTMENT_SPINE:
                learning_rate *= stdp_config->spine_scaling;
                break;
        }
        
        // Apply weight change
        float weight_change = total_plasticity * learning_rate * dt;
        float old_weight = synapse.weight;
        synapse.weight += weight_change;
        
        // Enforce weight bounds
        synapse.weight = fmaxf(synapse.min_weight, fminf(synapse.max_weight, synapse.weight));
        
        // Update plasticity history
        if (weight_change > 0) {
            plasticity.ltp_magnitude = weight_change;
            plasticity.potentiation_history += weight_change;
            plasticity.last_plasticity_event = current_time;
        } else if (weight_change < 0) {
            plasticity.ltd_magnitude = -weight_change;
            plasticity.depression_history += -weight_change;
            plasticity.last_plasticity_event = current_time;
        }
        
        // Update synaptic tag for late-phase plasticity
        if (fabs(weight_change) > 0.001f) {  // Significant change
            plasticity.synaptic_tag += fabs(weight_change);
        }
        
        // Update effective weight for immediate use
        synapse.effective_weight = synapse.weight * plasticity.scaling_factor;
    }
    
    // ========================================
    // HOMEOSTATIC SCALING
    // ========================================
    
    // Update activity integral for homeostasis
    float target_activity = HOMEOSTATIC_TARGET_RATE * dt * 0.001f;  // Convert Hz to probability
    float actual_activity = pre_spike ? 1.0f : 0.0f;
    
    float activity_decay = expf(-dt / HOMEOSTATIC_TIME_WINDOW);
    plasticity.activity_integral = plasticity.activity_integral * activity_decay + 
                                  actual_activity * dt;
    
    // Compute homeostatic scaling adjustment
    float activity_error = target_activity - plasticity.activity_integral;
    float scaling_adjustment = HOMEOSTATIC_SCALING_RATE * activity_error * dt;
    
    plasticity.scaling_factor += scaling_adjustment;
    plasticity.scaling_factor = fmaxf(0.1f, fminf(5.0f, plasticity.scaling_factor));
    
    // Apply homeostatic scaling to effective weight
    synapse.effective_weight = synapse.weight * plasticity.scaling_factor;
    
    // ========================================
    // STRUCTURAL PLASTICITY SIGNALS
    // ========================================
    
    // Update structural stability based on activity and strength
    float stability_factor = 1.0f + 0.1f * (synapse.weight / synapse.max_weight);
    stability_factor *= 1.0f + 0.1f * plasticity.activity_integral;
    
    plasticity.structural_stability += (stability_factor - plasticity.structural_stability) * 
                                      0.001f * dt;  // Slow adjustment
    
    // Compute growth and pruning signals
    if (plasticity.potentiation_history > plasticity.depression_history * 2.0f) {
        // Strong net potentiation - growth signal
        plasticity.growth_signal += 0.001f * dt;
    }
    
    if (plasticity.activity_integral < target_activity * 0.1f && 
        synapse.weight < synapse.max_weight * 0.1f) {
        // Very low activity and weak synapse - pruning signal
        plasticity.pruning_signal += 0.001f * dt;
    } else {
        // Decay pruning signal
        plasticity.pruning_signal *= expf(-dt / 86400000.0f);  // 24 hour decay
    }
    
    // Clamp structural signals
    plasticity.growth_signal = fmaxf(0.0f, fminf(1.0f, plasticity.growth_signal));
    plasticity.pruning_signal = fmaxf(0.0f, fminf(1.0f, plasticity.pruning_signal));
}

/**
 * Host function to launch enhanced STDP kernel
 */
void launchEnhancedSTDP(
    GPUSynapse* d_synapses,
    GPUNeuronState* d_neurons,
    PlasticityState* d_plasticity_states,
    STDPRuleConfig* d_stdp_config,
    float* d_global_neuromodulators,
    float current_time,
    float dt,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    enhancedSTDPKernel<<<grid, block>>>(
        d_synapses, d_neurons, d_plasticity_states, d_stdp_config,
        d_global_neuromodulators, current_time, dt, num_synapses
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in enhanced STDP: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

/**
 * Initialize STDP configuration with biologically realistic parameters
 */
STDPRuleConfig createDefaultSTDPConfig() {
    STDPRuleConfig config;
    
    // Timing-dependent parameters (based on Bi & Poo, 1998)
    config.pre_post_amp_positive = 0.005f;   // 0.5% weight change
    config.pre_post_amp_negative = 0.002f;   // 0.2% weight change
    config.post_pre_amp_positive = 0.002f;   // 0.2% weight change
    config.post_pre_amp_negative = 0.005f;   // 0.5% weight change
    
    // Time constants (ms)
    config.tau_positive = 20.0f;             // LTP time constant
    config.tau_negative = 20.0f;             // LTD time constant
    config.tau_x = 15.0f;                    // Pre-synaptic trace
    config.tau_y = 30.0f;                    // Post-synaptic trace
    
    // Calcium-dependent parameters
    config.ca_amplitude_factor = 2.0f;       // Calcium enhances plasticity
    config.ca_threshold_factor = 0.5f;       // Threshold modulation
    config.ca_cooperativity = 2.0f;          // Cooperative calcium binding
    
    // Frequency-dependent parameters
    config.frequency_factor = 1.5f;          // High frequency enhancement
    config.burst_factor = 2.0f;              // Burst enhancement
    config.pattern_sensitivity = 1.0f;       // Pattern recognition factor
    
    // Compartment-specific scaling
    config.soma_scaling = 0.8f;              // Reduced somatic plasticity
    config.basal_scaling = 1.0f;             // Standard basal plasticity
    config.apical_scaling = 1.5f;            // Enhanced apical plasticity
    config.spine_scaling = 2.0f;             // Enhanced spine plasticity
    
    // Receptor-specific scaling
    config.ampa_plasticity_factor = 1.0f;    // Standard AMPA plasticity
    config.nmda_plasticity_factor = 2.0f;    // Enhanced NMDA plasticity
    config.gaba_plasticity_factor = 0.5f;    // Reduced inhibitory plasticity
    
    // Learning rate parameters
    config.base_learning_rate = 0.001f;      // 0.1% base rate
    config.max_learning_rate = 0.01f;        // 1% maximum rate
    config.learning_rate_decay = 0.99f;      // Slow decay
    
    return config;
}

#endif // ENHANCED_STDP_FRAMEWORK_H