// AdvancedReinforcementLearning.h
#ifndef ADVANCED_REINFORCEMENT_LEARNING_H
#define ADVANCED_REINFORCEMENT_LEARNING_H

#include "GPUNeuralStructures.h"
#include "EnhancedSTDPFramework.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Advanced reinforcement learning framework implementing:
 * - Temporal Difference learning with multi-timescale value functions
 * - Dopaminergic reward prediction error (RPE) computation
 * - Actor-Critic architecture with eligibility traces
 * - Hierarchical reinforcement learning for complex behaviors
 * - Intrinsic motivation and curiosity-driven learning
 */

// ========================================
// REINFORCEMENT LEARNING CONSTANTS
// ========================================

// Value function parameters
#define VALUE_FUNCTION_DISCOUNT     0.95f     // Temporal discount factor (γ)
#define VALUE_LEARNING_RATE         0.01f     // Value function learning rate (α)
#define POLICY_LEARNING_RATE        0.001f    // Policy learning rate
#define ELIGIBILITY_DECAY_LAMBDA    0.9f      // Eligibility trace decay (λ)

// Dopamine system parameters
#define DOPAMINE_BASELINE           0.5f      // Baseline dopamine level
#define DOPAMINE_BURST_AMPLITUDE    2.0f      // Maximum burst amplitude
#define DOPAMINE_DIP_AMPLITUDE      0.1f      // Minimum dip amplitude
#define DOPAMINE_TIME_CONSTANT      100.0f    // Dopamine decay time constant (ms)
#define DOPAMINE_DIFFUSION_RATE     0.1f      // Spatial diffusion rate

// RPE computation parameters
#define RPE_INTEGRATION_WINDOW      500.0f    // ms - window for RPE integration
#define RPE_SURPRISE_THRESHOLD      0.2f      // Threshold for surprise detection
#define RPE_CONFIDENCE_FACTOR       0.8f      // Confidence in predictions

// Actor-Critic parameters
#define ACTOR_EXPLORATION_NOISE     0.1f      // Exploration noise level
#define CRITIC_REGULARIZATION       0.001f    // L2 regularization for critic
#define ADVANTAGE_NORMALIZATION     true      // Normalize advantage estimates

// Intrinsic motivation parameters
#define CURIOSITY_WEIGHT            0.1f      // Weight of curiosity reward
#define NOVELTY_DECAY_RATE          0.01f     // Decay rate for novelty
#define INFORMATION_GAIN_THRESHOLD  0.05f     // Threshold for information gain

/**
 * Value function representation for temporal difference learning
 */
struct ValueFunction {
    // Multi-timescale value estimates
    float short_term_value;        // Value estimate (100ms-1s)
    float medium_term_value;       // Value estimate (1s-10s) 
    float long_term_value;         // Value estimate (10s-100s)
    float very_long_term_value;    // Value estimate (minutes)
    
    // Eligibility traces for TD learning
    float value_eligibility;       // Eligibility trace for value updates
    float policy_eligibility;      // Eligibility trace for policy updates
    float feature_eligibility[16]; // Feature-specific eligibility traces
    
    // Prediction confidence and uncertainty
    float prediction_confidence;   // Confidence in current value estimate
    float prediction_uncertainty;  // Uncertainty measure (Bayesian)
    float surprise_accumulator;    // Accumulated surprise signal
    
    // Learning progress tracking
    float value_error_history;     // Historical value prediction errors
    float learning_progress;       // Rate of learning progress
    float competence_estimate;     // Estimated competence in this state
};

/**
 * Dopamine neuron model with realistic dynamics
 */
struct DopamineNeuron {
    // Neuron state variables
    float membrane_potential;      // Current membrane potential
    float firing_rate;            // Current firing rate (Hz)
    float baseline_activity;      // Baseline firing rate
    float burst_threshold;        // Threshold for burst firing
    
    // RPE computation variables
    float predicted_reward;       // Current reward prediction
    float actual_reward;          // Received reward signal
    float reward_prediction_error; // RPE = actual - predicted
    float rpe_history[10];        // Recent RPE history
    
    // Temporal dynamics
    float value_prediction;       // Current state value prediction
    float previous_value;         // Previous state value
    float td_error;              // Temporal difference error
    float td_error_filtered;     // Low-pass filtered TD error
    
    // Neuromodulator release
    float dopamine_concentration; // Local dopamine concentration
    float dopamine_release_rate;  // Current release rate
    float dopamine_uptake_rate;   // Reuptake rate
    float dopamine_diffusion;     // Spatial diffusion coefficient
    
    // Adaptation and learning
    float adaptation_level;       // Current adaptation state
    float learning_rate_modulation; // Dynamic learning rate
    float exploration_drive;      // Drive for exploration
    float exploitation_preference; // Preference for exploitation
};

/**
 * Actor-Critic architecture for policy learning
 */
struct ActorCriticState {
    // Actor (policy) components
    float action_preferences[32]; // Preferences for different actions
    float action_probabilities[32]; // Softmax action probabilities
    float policy_parameters[32];  // Learned policy parameters
    float action_eligibility[32]; // Action-specific eligibility traces
    
    // Critic (value function) components
    float state_value;           // Current state value estimate
    float state_features[64];    // State feature representation
    float value_weights[64];     // Value function weights
    float baseline_estimate;     // Baseline for advantage computation
    
    // Advantage computation
    float advantage_estimate;    // Current advantage (A = Q - V)
    float advantage_history[5];  // Recent advantage estimates
    float advantage_variance;    // Variance of advantage estimates
    
    // Exploration vs exploitation
    float exploration_bonus;     // Bonus for exploration
    float uncertainty_estimate;  // Epistemic uncertainty
    float information_gain;      // Expected information gain
    float novelty_signal;        // Novelty detection signal
    
    // Hierarchical learning
    float subgoal_preferences[16]; // Preferences for subgoals
    float temporal_abstraction;  // Level of temporal abstraction
    float goal_hierarchy_level;  // Current level in goal hierarchy
    
    // Meta-learning components
    float learning_to_learn_signal; // Meta-learning signal
    float adaptation_speed;      // Speed of adaptation to new tasks
    float transfer_potential;    // Potential for transfer learning
};

/**
 * Intrinsic motivation and curiosity system
 */
struct CuriositySystem {
    // Prediction model components
    float world_model_prediction[32]; // Predicted next state
    float prediction_error[32];   // Prediction error signal
    float prediction_confidence;  // Confidence in world model
    float model_uncertainty;      // Epistemic uncertainty in model
    
    // Novelty detection
    float novelty_detector[16];   // Novelty detection features
    float familiarity_level;      // How familiar current state is
    float surprise_level;         // Current surprise level
    float exploration_value;      // Value of exploring current state
    
    // Information theoretic measures
    float information_gain;       // Expected information gain
    float entropy_estimate;       // Entropy of current state distribution
    float mutual_information;     // Mutual information with past states
    float empowerment;           // Empowerment (future state reachability)
    
    // Competence and progress
    float competence_progress;    // Rate of competence improvement
    float mastery_level;         // Current mastery of environment
    float challenge_level;        // Perceived challenge level
    float flow_state;            // Flow state indicator (challenge vs skill)
    
    // Exploration strategies
    float random_exploration;     // Random exploration drive
    float directed_exploration;   // Directed exploration drive
    float social_exploration;     // Social/imitation learning drive
    float goal_exploration;       // Goal-directed exploration
};

/**
 * CUDA kernel for dopamine system and RPE computation
 */
__global__ void dopamineSystemKernel(
    DopamineNeuron* dopamine_neurons,
    ValueFunction* value_functions,
    ActorCriticState* actor_critic_states,
    GPUNeuronState* network_neurons,
    float* global_reward_signal,
    float* environmental_features,
    float current_time,
    float dt,
    int num_dopamine_neurons,
    int num_network_neurons
) {
    int da_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (da_idx >= num_dopamine_neurons) return;
    
    DopamineNeuron& da_neuron = dopamine_neurons[da_idx];
    ValueFunction& value_func = value_functions[da_idx];
    ActorCriticState& actor_critic = actor_critic_states[da_idx];
    
    // ========================================
    // VALUE FUNCTION UPDATE (CRITIC)
    // ========================================
    
    // Extract state features from network activity
    float network_activity = 0.0f;
    float network_synchrony = 0.0f;
    float network_complexity = 0.0f;
    
    // Sample network activity for state representation
    for (int i = 0; i < min(64, num_network_neurons); i++) {
        int neuron_idx = (da_idx * 64 + i) % num_network_neurons;
        GPUNeuronState& neuron = network_neurons[neuron_idx];
        
        if (neuron.active) {
            network_activity += neuron.activity_level;
            network_synchrony += cosf(neuron.voltage * 0.1f); // Phase synchrony
            network_complexity += fabs(neuron.total_excitatory_input - 
                                      neuron.total_inhibitory_input);
        }
    }
    
    // Normalize and store state features
    actor_critic.state_features[0] = network_activity / 64.0f;
    actor_critic.state_features[1] = network_synchrony / 64.0f;
    actor_critic.state_features[2] = network_complexity / 64.0f;
    
    // Add environmental features
    for (int i = 0; i < min(61, 32); i++) {
        actor_critic.state_features[i + 3] = environmental_features[i];
    }
    
    // Compute current state value using linear approximation
    float current_state_value = 0.0f;
    for (int i = 0; i < 64; i++) {
        current_state_value += actor_critic.state_features[i] * 
                              actor_critic.value_weights[i];
    }
    actor_critic.state_value = current_state_value;
    
    // ========================================
    // TEMPORAL DIFFERENCE ERROR COMPUTATION
    // ========================================
    
    // Get current reward signal
    float current_reward = global_reward_signal[0];
    
    // Compute TD error: δ = r + γV(s') - V(s)
    float td_error = current_reward + 
                    VALUE_FUNCTION_DISCOUNT * current_state_value - 
                    value_func.short_term_value;
    
    da_neuron.td_error = td_error;
    
    // Apply exponential smoothing to TD error
    da_neuron.td_error_filtered = da_neuron.td_error_filtered * 0.9f + 
                                 td_error * 0.1f;
    
    // ========================================
    // REWARD PREDICTION ERROR (RPE)
    // ========================================
    
    // Update reward prediction based on current state
    da_neuron.predicted_reward = current_state_value;
    da_neuron.actual_reward = current_reward;
    
    // Compute RPE with confidence weighting
    float prediction_confidence = fminf(1.0f, fmaxf(0.1f, 
        1.0f - fabs(da_neuron.td_error_filtered)));
    
    da_neuron.reward_prediction_error = (da_neuron.actual_reward - 
                                        da_neuron.predicted_reward) * 
                                       prediction_confidence;
    
    // Update RPE history (sliding window)
    for (int i = 9; i > 0; i--) {
        da_neuron.rpe_history[i] = da_neuron.rpe_history[i-1];
    }
    da_neuron.rpe_history[0] = da_neuron.reward_prediction_error;
    
    // ========================================
    // DOPAMINE DYNAMICS
    // ========================================
    
    // Compute dopamine firing rate based on RPE
    float rpe_magnitude = fabs(da_neuron.reward_prediction_error);
    float rpe_sign = (da_neuron.reward_prediction_error > 0) ? 1.0f : -1.0f;
    
    // Positive RPE increases firing, negative RPE decreases it
    if (da_neuron.reward_prediction_error > 0) {
        // Reward better than expected - dopamine burst
        da_neuron.firing_rate = da_neuron.baseline_activity + 
                               DOPAMINE_BURST_AMPLITUDE * rpe_magnitude;
    } else if (da_neuron.reward_prediction_error < -0.1f) {
        // Reward worse than expected - dopamine dip
        da_neuron.firing_rate = da_neuron.baseline_activity * 
                               (DOPAMINE_DIP_AMPLITUDE + 0.9f * expf(-5.0f * rpe_magnitude));
    } else {
        // Small prediction errors - return to baseline
        da_neuron.firing_rate += (da_neuron.baseline_activity - da_neuron.firing_rate) * 
                                 0.1f * dt;
    }
    
    // Update dopamine concentration based on firing rate
    float da_release = da_neuron.firing_rate * da_neuron.dopamine_release_rate * dt;
    float da_uptake = da_neuron.dopamine_concentration * da_neuron.dopamine_uptake_rate * dt;
    
    da_neuron.dopamine_concentration += da_release - da_uptake;
    da_neuron.dopamine_concentration = fmaxf(0.0f, fminf(5.0f, 
                                            da_neuron.dopamine_concentration));
    
    // ========================================
    // VALUE FUNCTION LEARNING
    // ========================================
    
    // Update value function weights using TD error
    float value_learning_rate = VALUE_LEARNING_RATE * 
        (1.0f + 0.5f * da_neuron.dopamine_concentration);
    
    for (int i = 0; i < 64; i++) {
        // Update eligibility trace
        value_func.value_eligibility = value_func.value_eligibility * 
            ELIGIBILITY_DECAY_LAMBDA + actor_critic.state_features[i];
        
        // Update weight
        actor_critic.value_weights[i] += value_learning_rate * td_error * 
                                        value_func.value_eligibility;
        
        // Apply regularization
        actor_critic.value_weights[i] *= (1.0f - CRITIC_REGULARIZATION * dt);
    }
    
    // Update multi-timescale value estimates
    value_func.short_term_value += 0.1f * dt * 
        (current_state_value - value_func.short_term_value);
    value_func.medium_term_value += 0.01f * dt * 
        (current_state_value - value_func.medium_term_value);
    value_func.long_term_value += 0.001f * dt * 
        (current_state_value - value_func.long_term_value);
    
    // ========================================
    // ACTOR (POLICY) UPDATE
    // ========================================
    
    // Compute advantage estimate
    actor_critic.advantage_estimate = td_error; // Simplified TD error as advantage
    
    // Apply advantage normalization if enabled
    if (ADVANTAGE_NORMALIZATION) {
        // Update advantage statistics
        float advantage_mean = 0.0f;
        for (int i = 0; i < 5; i++) {
            advantage_mean += actor_critic.advantage_history[i];
        }
        advantage_mean /= 5.0f;
        
        float advantage_var = 0.0f;
        for (int i = 0; i < 5; i++) {
            float diff = actor_critic.advantage_history[i] - advantage_mean;
            advantage_var += diff * diff;
        }
        advantage_var /= 5.0f;
        
        // Normalize advantage
        if (advantage_var > 1e-6f) {
            actor_critic.advantage_estimate = (actor_critic.advantage_estimate - advantage_mean) / 
                                             sqrtf(advantage_var + 1e-6f);
        }
    }
    
    // Update advantage history
    for (int i = 4; i > 0; i--) {
        actor_critic.advantage_history[i] = actor_critic.advantage_history[i-1];
    }
    actor_critic.advantage_history[0] = actor_critic.advantage_estimate;
    
    // Policy gradient update (simplified)
    float policy_learning_rate = POLICY_LEARNING_RATE * 
        (1.0f + da_neuron.dopamine_concentration);
    
    for (int a = 0; a < 32; a++) {
        // Update action eligibility
        actor_critic.action_eligibility[a] *= ELIGIBILITY_DECAY_LAMBDA;
        
        // Add current action probability to eligibility
        actor_critic.action_eligibility[a] += actor_critic.action_probabilities[a];
        
        // Policy gradient update
        actor_critic.policy_parameters[a] += policy_learning_rate * 
            actor_critic.advantage_estimate * actor_critic.action_eligibility[a];
    }
    
    // Compute action probabilities using softmax
    float exp_sum = 0.0f;
    for (int a = 0; a < 32; a++) {
        actor_critic.action_preferences[a] = expf(actor_critic.policy_parameters[a]);
        exp_sum += actor_critic.action_preferences[a];
    }
    
    for (int a = 0; a < 32; a++) {
        actor_critic.action_probabilities[a] = actor_critic.action_preferences[a] / 
                                              (exp_sum + 1e-8f);
    }
    
    // ========================================
    // EXPLORATION VS EXPLOITATION
    // ========================================
    
    // Update exploration drive based on prediction uncertainty
    float prediction_uncertainty = rpe_magnitude / (1.0f + rpe_magnitude);
    actor_critic.uncertainty_estimate = actor_critic.uncertainty_estimate * 0.99f + 
                                       prediction_uncertainty * 0.01f;
    
    // Compute exploration bonus
    actor_critic.exploration_bonus = ACTOR_EXPLORATION_NOISE * 
        sqrtf(actor_critic.uncertainty_estimate);
    
    // Update exploration vs exploitation balance
    da_neuron.exploration_drive = fminf(1.0f, actor_critic.uncertainty_estimate + 
                                       actor_critic.exploration_bonus);
    da_neuron.exploitation_preference = 1.0f - da_neuron.exploration_drive;
}

/**
 * CUDA kernel for intrinsic motivation and curiosity
 */
__global__ void curiositySystemKernel(
    CuriositySystem* curiosity_systems,
    ActorCriticState* actor_critic_states,
    GPUNeuronState* network_neurons,
    float* environmental_features,
    float current_time,
    float dt,
    int num_systems,
    int num_network_neurons
) {
    int sys_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys_idx >= num_systems) return;
    
    CuriositySystem& curiosity = curiosity_systems[sys_idx];
    ActorCriticState& actor_critic = actor_critic_states[sys_idx];
    
    // ========================================
    // WORLD MODEL PREDICTION
    // ========================================
    
    // Predict next state based on current state and action
    // This is a simplified linear prediction model
    for (int i = 0; i < 32; i++) {
        float prediction = 0.0f;
        
        // Combine current state features and action probabilities
        for (int j = 0; j < min(32, 64); j++) {
            prediction += actor_critic.state_features[j] * 0.1f;
        }
        for (int j = 0; j < 32; j++) {
            prediction += actor_critic.action_probabilities[j] * 0.05f;
        }
        
        curiosity.world_model_prediction[i] = tanhf(prediction);
    }
    
    // ========================================
    // PREDICTION ERROR AND SURPRISE
    // ========================================
    
    // Compute prediction error by comparing with actual next state
    float total_prediction_error = 0.0f;
    for (int i = 0; i < 32; i++) {
        float actual_feature = (i < 32) ? environmental_features[i] : 0.0f;
        curiosity.prediction_error[i] = actual_feature - curiosity.world_model_prediction[i];
        total_prediction_error += curiosity.prediction_error[i] * curiosity.prediction_error[i];
    }
    
    // Update surprise level
    float current_surprise = sqrtf(total_prediction_error / 32.0f);
    curiosity.surprise_level = curiosity.surprise_level * 0.9f + current_surprise * 0.1f;
    
    // ========================================
    // NOVELTY DETECTION
    // ========================================
    
    // Simple novelty detection based on state uniqueness
    float state_novelty = 0.0f;
    for (int i = 0; i < 16; i++) {
        // Compute distance from previous states (simplified)
        float feature = (i < 32) ? environmental_features[i] : 0.0f;
        float distance = fabs(feature - curiosity.novelty_detector[i]);
        state_novelty += distance;
        
        // Update novelty detector with exponential moving average
        curiosity.novelty_detector[i] = curiosity.novelty_detector[i] * 0.99f + 
                                       feature * 0.01f;
    }
    
    curiosity.familiarity_level = 1.0f / (1.0f + state_novelty);
    
    // ========================================
    // INFORMATION GAIN COMPUTATION
    // ========================================
    
    // Compute expected information gain from actions
    float entropy_before = 0.0f;
    float entropy_after = 0.0f;
    
    for (int i = 0; i < 32; i++) {
        float p = actor_critic.action_probabilities[i] + 1e-8f;
        entropy_before -= p * log2f(p);
        
        // Estimated entropy after taking action (simplified)
        float p_after = p * (1.0f + curiosity.surprise_level * 0.1f);
        p_after = fminf(1.0f, p_after);
        entropy_after -= p_after * log2f(p_after);
    }
    
    curiosity.information_gain = fmaxf(0.0f, entropy_after - entropy_before);
    
    // ========================================
    // COMPETENCE AND PROGRESS TRACKING
    // ========================================
    
    // Track learning progress as reduction in prediction error
    float learning_rate = 0.001f;
    float progress_signal = 0.0f;
    
    if (curiosity.surprise_level > 0.0f) {
        progress_signal = -learning_rate * curiosity.surprise_level; // Negative because we want to reduce error
    }
    
    curiosity.competence_progress = curiosity.competence_progress * 0.99f + 
                                   progress_signal * 0.01f;
    
    // Update mastery level based on prediction accuracy
    float prediction_accuracy = 1.0f / (1.0f + curiosity.surprise_level);
    curiosity.mastery_level = curiosity.mastery_level * 0.999f + 
                             prediction_accuracy * 0.001f;
    
    // ========================================
    // EXPLORATION DRIVE COMPUTATION
    // ========================================
    
    // Combine different exploration motivations
    curiosity.random_exploration = ACTOR_EXPLORATION_NOISE * 
        (1.0f - curiosity.mastery_level);
    
    curiosity.directed_exploration = curiosity.information_gain * 
        (1.0f - curiosity.familiarity_level);
    
    curiosity.goal_exploration = (curiosity.surprise_level > RPE_SURPRISE_THRESHOLD) ? 
        curiosity.surprise_level * 0.5f : 0.0f;
    
    // Total exploration value
    curiosity.exploration_value = curiosity.random_exploration + 
                                 curiosity.directed_exploration + 
                                 curiosity.goal_exploration;
    
    // ========================================
    // INTRINSIC REWARD GENERATION
    // ========================================
    
    // Generate intrinsic reward based on curiosity and learning progress
    float intrinsic_reward = CURIOSITY_WEIGHT * 
        (curiosity.information_gain + curiosity.competence_progress * 0.5f);
    
    // Add intrinsic reward to global reward signal (simplified - would need atomic operations)
    // This would typically be done in a separate reduction kernel
    
    // ========================================
    // FLOW STATE DETECTION
    // ========================================
    
    // Detect flow state as balance between challenge and skill
    curiosity.challenge_level = curiosity.surprise_level;
    float skill_level = curiosity.mastery_level;
    
    // Flow occurs when challenge matches skill level
    float challenge_skill_ratio = curiosity.challenge_level / (skill_level + 1e-6f);
    curiosity.flow_state = 1.0f / (1.0f + fabs(challenge_skill_ratio - 1.0f));
    
    // Flow state enhances learning and exploration
    if (curiosity.flow_state > 0.7f) {
        // In flow - increase learning rate and reduce random exploration
        actor_critic.exploration_bonus *= 0.8f;
        curiosity.directed_exploration *= 1.2f;
    }
}

/**
 * Host function to launch reinforcement learning kernels
 */
void launchAdvancedReinforcementLearning(
    DopamineNeuron* d_dopamine_neurons,
    ValueFunction* d_value_functions,
    ActorCriticState* d_actor_critic_states,
    CuriositySystem* d_curiosity_systems,
    GPUNeuronState* d_network_neurons,
    float* d_global_reward_signal,
    float* d_environmental_features,
    float current_time,
    float dt,
    int num_dopamine_neurons,
    int num_network_neurons
) {
    // Launch dopamine system kernel
    {
        dim3 block(256);
        dim3 grid((num_dopamine_neurons + block.x - 1) / block.x);
        
        dopamineSystemKernel<<<grid, block>>>(
            d_dopamine_neurons, d_value_functions, d_actor_critic_states, 
            d_network_neurons, d_global_reward_signal, d_environmental_features,
            current_time, dt, num_dopamine_neurons, num_network_neurons
        );
    }
    
    // Launch curiosity system kernel
    {
        dim3 block(256);
        dim3 grid((num_dopamine_neurons + block.x - 1) / block.x);
        
        curiositySystemKernel<<<grid, block>>>(
            d_curiosity_systems, d_actor_critic_states, d_network_neurons,
            d_environmental_features, current_time, dt, 
            num_dopamine_neurons, num_network_neurons
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in reinforcement learning: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

#endif // ADVANCED_REINFORCEMENT_LEARNING_H