#ifndef HEBBIAN_LEARNING_KERNEL_CUH
#define HEBBIAN_LEARNING_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * Core Hebbian learning kernel implementing "cells that fire together, wire together"
 * This provides activity-dependent strengthening complementary to STDP timing rules
 */
__global__ void hebbianLearningKernel(GPUSynapse* synapses,
                                     const GPUNeuronState* neurons,
                                     float current_time,
                                     float dt,
                                     int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Validate neuron indices
    if (pre_idx < 0 || post_idx < 0) return;
    
    // ========================================
    // EXTRACT NEURAL ACTIVITIES
    // ========================================
    
    float pre_activity = neurons[pre_idx].activity_level;
    float post_activity = neurons[post_idx].activity_level;
    
    // Skip if either neuron is below minimum activity threshold
    if (pre_activity < MIN_ACTIVITY_THRESHOLD || post_activity < MIN_ACTIVITY_THRESHOLD) {
        return;
    }
    
    // ========================================
    // COMPUTE ACTIVITY CORRELATION
    // ========================================
    
    // Basic Hebbian correlation: product of pre and post activities
    float activity_correlation = pre_activity * post_activity;
    
    // Normalize by maximum possible correlation to prevent runaway growth
    float max_correlation = 1.0f;  // Assuming activities are normalized to [0,1]
    float normalized_correlation = activity_correlation / max_correlation;
    
    // Apply correlation threshold
    if (normalized_correlation < CORRELATION_THRESHOLD) return;
    
    // ========================================
    // COVARIANCE-BASED HEBBIAN RULE
    // ========================================
    
    // Implement the covariance rule: Δw ∝ (x_pre - <x_pre>)(x_post - <x_post>)
    // This prevents runaway potentiation by subtracting mean activities
    
    // Estimate running averages of neural activities
    // These should be computed elsewhere and stored in neuron state
    float pre_mean = neurons[pre_idx].average_activity;
    float post_mean = neurons[post_idx].average_activity;
    
    // Covariance-based correlation
    float pre_deviation = pre_activity - pre_mean;
    float post_deviation = post_activity - post_mean;
    float covariance_correlation = pre_deviation * post_deviation;
    
    // ========================================
    // SYNAPSE-TYPE SPECIFIC RULES
    // ========================================
    
    bool is_excitatory = synapse.weight > 0.0f;
    float hebbian_rate = HEBBIAN_LEARNING_RATE;
    
    // Different plasticity rules for excitatory vs inhibitory synapses
    if (is_excitatory) {
        // Excitatory synapses: standard Hebbian strengthening
        // But prevent excessive potentiation with saturation
        float weight_factor = 1.0f - (fabsf(synapse.weight) / MAX_WEIGHT);
        hebbian_rate *= weight_factor * weight_factor;  // Quadratic saturation
    } else {
        // Inhibitory synapses: modified anti-Hebbian rule
        // High correlation can actually weaken inhibitory connections
        // to maintain excitation-inhibition balance
        hebbian_rate *= -0.5f;
        covariance_correlation *= -1.0f;  // Invert for anti-Hebbian effect
    }
    
    // ========================================
    // COMPARTMENT-SPECIFIC MODULATION
    // ========================================
    
    int compartment = synapse.post_compartment;
    int compartment_type = (compartment == 0) ? COMPARTMENT_SOMA : 
                          (compartment <= 3) ? COMPARTMENT_BASAL : COMPARTMENT_APICAL;
    
    float compartment_factor = 1.0f;
    switch (compartment_type) {
        case COMPARTMENT_APICAL:
            // Apical dendrites: enhanced Hebbian learning for contextual associations
            compartment_factor = 1.2f;
            break;
        case COMPARTMENT_BASAL:
            // Basal dendrites: standard Hebbian learning for input associations
            compartment_factor = 1.0f;
            break;
        case COMPARTMENT_SOMA:
            // Somatic synapses: reduced Hebbian to prevent interference with spike generation
            compartment_factor = 0.7f;
            break;
    }
    
    hebbian_rate *= compartment_factor;
    
    // ========================================
    // RECEPTOR-TYPE MODULATION
    // ========================================
    
    float receptor_factor = 1.0f;
    switch (synapse.receptor_index) {
        case RECEPTOR_AMPA:
            // AMPA: fast, strong Hebbian plasticity
            receptor_factor = 1.0f;
            break;
        case RECEPTOR_NMDA:
            // NMDA: voltage-dependent Hebbian plasticity (simplified)
            // In reality, this would depend on postsynaptic voltage
            receptor_factor = 1.3f;
            break;
        case RECEPTOR_GABA_A:
            // GABA-A: anti-Hebbian plasticity for inhibitory balance
            receptor_factor = -0.5f;
            break;
        case RECEPTOR_GABA_B:
            // GABA-B: slower, modulatory anti-Hebbian plasticity
            receptor_factor = -0.3f;
            break;
    }
    
    hebbian_rate *= receptor_factor;
    
    // ========================================
    // ACTIVITY-DEPENDENT SCALING
    // ========================================
    
    // Scale learning rate by overall activity level
    float activity_scale = (pre_activity + post_activity) * 0.5f * ACTIVITY_SCALING_FACTOR;
    hebbian_rate *= activity_scale;
    
    // ========================================
    // COMPUTE HEBBIAN WEIGHT CHANGE
    // ========================================
    
    // Primary Hebbian term based on covariance
    float dw_hebbian = hebbian_rate * covariance_correlation * dt;
    
    // Add pure correlation term with smaller weight
    float dw_correlation = hebbian_rate * 0.3f * normalized_correlation * dt;
    
    // Total Hebbian change
    float total_dw = dw_hebbian + dw_correlation;
    
    // ========================================
    // METAPLASTIC MODULATION
    // ========================================
    
    // Recent synaptic activity affects Hebbian plasticity
    float meta_factor = 1.0f;
    if (synapse.recent_activity > META_THRESHOLD_HIGH) {
        // High recent activity: reduce Hebbian plasticity (saturation)
        meta_factor = 1.0f / (1.0f + (synapse.recent_activity - META_THRESHOLD_HIGH));
    } else if (synapse.recent_activity < META_THRESHOLD_LOW) {
        // Low recent activity: enhance Hebbian plasticity (homeostatic upscaling)
        meta_factor = 1.0f + (META_THRESHOLD_LOW - synapse.recent_activity) * 0.5f;
    }
    
    total_dw *= meta_factor;
    
    // ========================================
    // HETEROSYNAPTIC COMPETITION
    // ========================================
    
    // Implement competitive Hebbian learning where strengthening of one synapse
    // can lead to weakening of nearby synapses (simplified version)
    
    // This would require knowledge of neighboring synapses, which is complex on GPU
    // For now, implement via activity normalization
    float competition_factor = 1.0f;
    
    // If this synapse is very active relative to others on the same postsynaptic neuron,
    // reduce its Hebbian plasticity to allow others to compete
    float relative_activity = synapse.activity_metric / (neurons[post_idx].activity_level + EPSILON);
    if (relative_activity > 2.0f) {
        competition_factor = 1.0f / relative_activity;
    }
    
    total_dw *= competition_factor;
    
    // ========================================
    // APPLY WEIGHT CHANGES WITH CONSTRAINTS
    // ========================================
    
    // Apply Hebbian weight change
    synapse.weight += total_dw * synapse.plasticity_rate;
    
    // Enforce hard bounds
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Maintain synapse type consistency
    if (is_excitatory && synapse.weight < 0.0f) {
        synapse.weight = 0.0f;
    } else if (!is_excitatory && synapse.weight > 0.0f) {
        synapse.weight = 0.0f;
    }
    
    // ========================================
    // UPDATE SYNAPSE STATE
    // ========================================
    
    // Update activity metric
    synapse.activity_metric += fabsf(total_dw) * 0.1f;
    
    // Update recent activity
    synapse.recent_activity += fabsf(total_dw) * 0.05f;
    
    // Update last active time if significant change occurred
    if (fabsf(total_dw) > MIN_ACTIVITY_THRESHOLD) {
        synapse.last_active = current_time;
    }
}

/**
 * Oja's learning rule kernel for principal component analysis-like learning
 * Implements normalized Hebbian learning that maintains bounded weights
 */
__global__ void ojasLearningKernel(GPUSynapse* synapses,
                                  const GPUNeuronState* neurons,
                                  float learning_rate,
                                  float dt,
                                  int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx < 0 || post_idx < 0) return;
    
    // ========================================
    // EXTRACT ACTIVITIES
    // ========================================
    
    float x_pre = neurons[pre_idx].activity_level;
    float y_post = neurons[post_idx].activity_level;
    
    // Skip if insufficient activity
    if (x_pre < MIN_ACTIVITY_THRESHOLD || y_post < MIN_ACTIVITY_THRESHOLD) {
        return;
    }
    
    // ========================================
    // OJA'S RULE: Δw = η * y * (x - y * w)
    // ========================================
    
    float current_weight = synapse.weight;
    
    // Oja's update term
    float oja_term = y_post * (x_pre - y_post * current_weight);
    
    // Weight change
    float dw_oja = learning_rate * oja_term * dt;
    
    // ========================================
    // APPLY UPDATE WITH CONSTRAINTS
    // ========================================
    
    synapse.weight += dw_oja * synapse.plasticity_rate;
    
    // Oja's rule naturally bounds weights, but add safety constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Update activity metrics
    synapse.activity_metric += fabsf(dw_oja) * 0.1f;
}

/**
 * BCM (Bienenstock-Cooper-Munro) learning rule kernel
 * Implements sliding threshold for bidirectional plasticity
 */
__global__ void bcmLearningKernel(GPUSynapse* synapses,
                                 GPUNeuronState* neurons,
                                 float learning_rate,
                                 float dt,
                                 int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx < 0 || post_idx < 0) return;
    
    // ========================================
    // BCM PLASTICITY FUNCTION
    // ========================================
    
    float x_pre = neurons[pre_idx].activity_level;
    float y_post = neurons[post_idx].activity_level;
    
    // BCM threshold (stored in neuron state)
    float theta = neurons[post_idx].plasticity_threshold;
    
    // Update sliding threshold based on postsynaptic activity history
    float theta_decay = expf(-dt / 10000.0f);  // 10-second time constant
    neurons[post_idx].plasticity_threshold *= theta_decay;
    neurons[post_idx].plasticity_threshold += y_post * y_post * (1.0f - theta_decay);
    
    // BCM plasticity function: φ(y) = y(y - θ)
    float phi_y = y_post * (y_post - theta);
    
    // Weight change: Δw = η * φ(y) * x
    float dw_bcm = learning_rate * phi_y * x_pre * dt;
    
    // ========================================
    // APPLY BCM UPDATE
    // ========================================
    
    synapse.weight += dw_bcm * synapse.plasticity_rate;
    
    // Apply constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Update metrics
    synapse.activity_metric += fabsf(dw_bcm) * 0.1f;
}

/**
 * Correlation-based learning kernel for detecting statistical dependencies
 */
__global__ void correlationLearningKernel(GPUSynapse* synapses,
                                         const GPUNeuronState* neurons,
                                         float* correlation_matrix,
                                         float learning_rate,
                                         float dt,
                                         int num_synapses,
                                         int matrix_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx < 0 || post_idx < 0 || pre_idx >= matrix_size || post_idx >= matrix_size) {
        return;
    }
    
    // ========================================
    // UPDATE CORRELATION MATRIX
    // ========================================
    
    float pre_activity = neurons[pre_idx].activity_level;
    float post_activity = neurons[post_idx].activity_level;
    
    // Update running correlation estimate
    int correlation_idx = pre_idx * matrix_size + post_idx;
    
    float current_correlation = correlation_matrix[correlation_idx];
    float new_correlation = pre_activity * post_activity;
    
    // Exponential moving average
    float alpha = 0.01f * dt;  // Learning rate for correlation estimate
    correlation_matrix[correlation_idx] = (1.0f - alpha) * current_correlation + 
                                         alpha * new_correlation;
    
    // ========================================
    // CORRELATION-BASED WEIGHT UPDATE
    // ========================================
    
    float correlation_strength = correlation_matrix[correlation_idx];
    
    // Weight change proportional to correlation strength
    float dw_corr = learning_rate * correlation_strength * dt;
    
    // Apply update
    synapse.weight += dw_corr * synapse.plasticity_rate;
    
    // Apply constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
}

#endif // HEBBIAN_LEARNING_KERNEL_CUH