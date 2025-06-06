#ifndef REWARD_MODULATION_KERNEL_CUH
#define REWARD_MODULATION_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * Reward prediction error computation kernel
 * Implements temporal difference learning for dopaminergic signaling
 */
__global__ void rewardPredictionErrorKernel(const GPUNeuronState* neurons,
                                           float actual_reward,
                                           float* predicted_reward,
                                           float* prediction_error,
                                           float* dopamine_level,
                                           float current_time,
                                           float dt,
                                           int num_neurons) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 computes global prediction error
    if (idx != 0) return;
    
    // ========================================
    // COMPUTE REWARD PREDICTION
    // ========================================
    
    float total_prediction = 0.0f;
    int prediction_neuron_count = 0;
    
    // Aggregate predictions from reward-predicting neurons
    for (int i = 0; i < num_neurons; i++) {
        if (neurons[i].neuron_type == NEURON_REWARD_PREDICTION && neurons[i].spiked) {
            // Weight prediction by neuron's activity level and recent firing
            float neuron_contribution = neurons[i].activity_level * 
                                       expf(-(current_time - neurons[i].last_spike_time) / 100.0f);
            total_prediction += neuron_contribution;
            prediction_neuron_count++;
        }
    }
    
    // Normalize prediction
    if (prediction_neuron_count > 0) {
        *predicted_reward = total_prediction / prediction_neuron_count;
    } else {
        *predicted_reward = 0.0f;
    }
    
    // ========================================
    // COMPUTE PREDICTION ERROR (RPE)
    // ========================================
    
    // Temporal difference error: δ = r + γV(s') - V(s)
    // Simplified here as: δ = actual_reward - predicted_reward
    *prediction_error = actual_reward - *predicted_reward;
    
    // Apply scaling to make prediction errors biologically realistic
    *prediction_error *= PREDICTION_ERROR_SCALE;
    
    // ========================================
    // UPDATE DOPAMINE LEVEL
    // ========================================
    
    // Dopamine level represents the modulatory signal strength
    // It follows the prediction error with some dynamics
    float dopamine_decay = expf(-dt / DOPAMINE_TAU);
    *dopamine_level *= dopamine_decay;
    
    // Dopamine response to prediction error (phasic response)
    if (*prediction_error > 0.0f) {
        // Positive prediction error: increase dopamine
        *dopamine_level += *prediction_error * 0.5f;
    } else {
        // Negative prediction error: decrease dopamine
        *dopamine_level += *prediction_error * 0.3f;  // Asymmetric response
    }
    
    // Add baseline dopamine level
    *dopamine_level += BASELINE_DOPAMINE;
    
    // Clamp dopamine level to physiological range
    *dopamine_level = fmaxf(-2.0f, fminf(3.0f, *dopamine_level));
}

/**
 * Main reward modulation kernel
 * Applies dopaminergic modulation to synaptic plasticity via eligibility traces
 */
__global__ void rewardModulationKernel(GPUSynapse* synapses,
                                      const GPUNeuronState* neurons,
                                      float reward_signal,
                                      float dopamine_level,
                                      float prediction_error,
                                      float current_time,
                                      float dt,
                                      int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    // Skip if no significant modulation signal
    if (fabsf(dopamine_level - BASELINE_DOPAMINE) < MIN_REWARD_THRESHOLD) return;
    
    // ========================================
    // COMPUTE EFFECTIVE REWARD SIGNAL
    // ========================================
    
    // Combine external reward with internal dopaminergic signal
    float effective_reward = reward_signal + (dopamine_level - BASELINE_DOPAMINE);
    
    // Apply temporal discounting for delayed rewards
    float time_since_last_activity = current_time - synapse.last_active;
    float discount_factor = expf(-time_since_last_activity / (REWARD_PREDICTION_TAU * 2.0f));
    effective_reward *= discount_factor;
    
    // ========================================
    // SYNAPSE-SPECIFIC DOPAMINE SENSITIVITY
    // ========================================
    
    // Different synapses have different sensitivity to dopamine
    float dopamine_sensitivity = synapse.dopamine_sensitivity;
    if (dopamine_sensitivity <= 0.0f) {
        dopamine_sensitivity = 1.0f;  // Default sensitivity
    }
    
    // Modulate sensitivity based on synapse type and location
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    int compartment = synapse.post_compartment;
    
    if (pre_idx >= 0 && post_idx >= 0) {
        // Dopaminergic modulation is stronger for certain pathway types
        int pre_type = neurons[pre_idx].neuron_type;
        int post_type = neurons[post_idx].neuron_type;
        
        if (pre_type == NEURON_EXCITATORY && post_type == NEURON_EXCITATORY) {
            dopamine_sensitivity *= 1.2f;  // Enhanced for excitatory-excitatory connections
        } else if (pre_type == NEURON_INHIBITORY) {
            dopamine_sensitivity *= 0.8f;  // Reduced for inhibitory connections
        }
        
        // Compartment-specific modulation
        int compartment_type = (compartment == 0) ? COMPARTMENT_SOMA : 
                              (compartment <= 3) ? COMPARTMENT_BASAL : COMPARTMENT_APICAL;
        
        if (compartment_type == COMPARTMENT_APICAL) {
            dopamine_sensitivity *= 1.3f;  // Enhanced apical modulation (top-down signals)
        }
    }
    
    // ========================================
    // TRACE-SPECIFIC MODULATION
    // ========================================
    
    // Different eligibility traces respond differently to dopamine
    float modulated_fast = synapse.fast_trace * effective_reward * 
                          FAST_TRACE_DOPAMINE_SENS * dopamine_sensitivity;
    
    float modulated_medium = synapse.medium_trace * effective_reward * 
                            MEDIUM_TRACE_DOPAMINE_SENS * dopamine_sensitivity;
    
    float modulated_slow = synapse.slow_trace * effective_reward * 
                          SLOW_TRACE_DOPAMINE_SENS * dopamine_sensitivity;
    
    // ========================================
    // COMPUTE REWARD-MODULATED WEIGHT CHANGES
    // ========================================
    
    // Fast component: immediate reward-dependent plasticity
    float dw_fast = modulated_fast * dt;
    
    // Medium component: short-term memory consolidation
    float dw_medium = modulated_medium * dt * 0.5f;
    
    // Slow component: long-term memory maintenance
    float dw_slow = modulated_slow * dt * 0.1f;
    
    // Total reward-modulated weight change
    float total_dw = dw_fast + dw_medium + dw_slow;
    
    // ========================================
    // PREDICTION ERROR-SPECIFIC MODULATION
    // ========================================
    
    // Prediction errors drive different types of plasticity
    if (prediction_error > 0.0f) {
        // Positive prediction error: unexpected reward
        // Strengthen connections that led to this outcome
        total_dw *= (1.0f + fabsf(prediction_error) * 0.5f);
        
    } else if (prediction_error < 0.0f) {
        // Negative prediction error: expected reward didn't occur
        // Weaken connections that led to false predictions
        total_dw *= (1.0f - fabsf(prediction_error) * 0.3f);
    }
    
    // ========================================
    // LATE-PHASE PLASTICITY TRIGGERING
    // ========================================
    
    // Strong reward signals can trigger protein synthesis-dependent plasticity
    if (fabsf(effective_reward) > PROTEIN_SYNTHESIS_THRESHOLD &&
        fabsf(synapse.tag_strength) > TAG_THRESHOLD) {
        
        // Check if tag and reward have compatible signs
        bool signs_match = (synapse.tag_strength > 0.0f && effective_reward > 0.0f) ||
                          (synapse.tag_strength < 0.0f && effective_reward < 0.0f);
        
        if (signs_match) {
            // Trigger late-phase plasticity
            float late_phase_dw = synapse.tag_strength * effective_reward * 
                                 LATE_PHASE_FACTOR * dopamine_sensitivity;
            
            total_dw += late_phase_dw;
            
            // Partially consume the tag
            synapse.tag_strength *= 0.7f;
        }
    }
    
    // ========================================
    // METAPLASTIC MODULATION
    // ========================================
    
    // Recent activity affects sensitivity to reward modulation
    float activity_factor = 1.0f;
    if (synapse.recent_activity > META_THRESHOLD_HIGH) {
        // High recent activity: reduced reward sensitivity (homeostatic)
        activity_factor = 1.0f / (1.0f + synapse.recent_activity);
    } else if (synapse.recent_activity < META_THRESHOLD_LOW) {
        // Low recent activity: increased reward sensitivity
        activity_factor = 1.0f + (META_THRESHOLD_LOW - synapse.recent_activity);
    }
    
    total_dw *= activity_factor;
    
    // ========================================
    // APPLY WEIGHT CHANGES WITH CONSTRAINTS
    // ========================================
    
    // Apply the reward-modulated weight change
    synapse.weight += total_dw * synapse.plasticity_rate;
    
    // Enforce weight bounds
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Maintain synapse type consistency
    bool is_excitatory = synapse.type == NEURON_EXCITATORY;
    if (is_excitatory && synapse.weight < 0.0f) {
        synapse.weight = 0.0f;
    } else if (!is_excitatory && synapse.weight > 0.0f) {
        synapse.weight = 0.0f;
    }
    
    // ========================================
    // UPDATE MODULATION STATE
    // ========================================
    
    // Update plasticity modulation level for this synapse
    synapse.plasticity_modulation = dopamine_level * dopamine_sensitivity;
    
    // Track cumulative reward modulation
    synapse.activity_metric += fabsf(total_dw) * 0.1f;
    
    // Update last active time if significant change occurred
    if (fabsf(total_dw) > MIN_REWARD_THRESHOLD) {
        synapse.last_active = current_time;
    }
}

/**
 * Dopamine sensitivity adaptation kernel
 * Adjusts individual synapse sensitivity to dopaminergic modulation
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   const GPUNeuronState* neurons,
                                                   float average_reward_history,
                                                   float current_time,
                                                   float dt,
                                                   int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    // ========================================
    // ADAPTIVE DOPAMINE SENSITIVITY
    // ========================================
    
    // Synapses adapt their dopamine sensitivity based on reward history
    // This implements a form of metaplasticity for reward learning
    
    float target_sensitivity = 1.0f;
    
    // If average rewards are consistently high, reduce sensitivity
    if (average_reward_history > 0.5f) {
        target_sensitivity = 1.0f / (1.0f + average_reward_history);
    }
    // If average rewards are consistently low, increase sensitivity
    else if (average_reward_history < -0.5f) {
        target_sensitivity = 1.0f + fabsf(average_reward_history);
    }
    
    // Slowly adapt sensitivity toward target
    float adaptation_rate = 0.001f * dt;
    synapse.dopamine_sensitivity += (target_sensitivity - synapse.dopamine_sensitivity) * 
                                   adaptation_rate;
    
    // Clamp sensitivity to reasonable bounds
    synapse.dopamine_sensitivity = fmaxf(0.1f, fminf(3.0f, synapse.dopamine_sensitivity));
    
    // ========================================
    // SYNAPTIC COMPETITION FOR DOPAMINE
    // ========================================
    
    // Implement competitive dynamics where active synapses compete for dopaminergic resources
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx >= 0 && post_idx >= 0) {
        float pre_activity = neurons[pre_idx].activity_level;
        float post_activity = neurons[post_idx].activity_level;
        
        // Synapses between active neurons get enhanced dopamine sensitivity
        float activity_boost = (pre_activity * post_activity) * 0.1f;
        synapse.dopamine_sensitivity += activity_boost * dt;
        
        // But prevent runaway increases
        if (synapse.dopamine_sensitivity > 2.0f) {
            synapse.dopamine_sensitivity = 2.0f;
        }
    }
}

/**
 * Reward trace update kernel for temporal credit assignment
 * Maintains a reward eligibility trace at the network level
 */
__global__ void rewardTraceUpdateKernel(float* network_reward_trace,
                                       float current_reward,
                                       float dt) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;  // Only thread 0 updates global trace
    
    // Decay existing reward trace
    float decay = expf(-dt / REWARD_PREDICTION_TAU);
    *network_reward_trace *= decay;
    
    // Add current reward
    *network_reward_trace += current_reward;
    
    // Apply bounds
    *network_reward_trace = fmaxf(-5.0f, fminf(5.0f, *network_reward_trace));
}

#endif // REWARD_MODULATION_KERNEL_CUH