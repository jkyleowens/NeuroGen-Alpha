#ifndef REWARD_MODULATION_KERNEL_CUH
#define REWARD_MODULATION_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

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