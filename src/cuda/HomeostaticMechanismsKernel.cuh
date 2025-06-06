#ifndef HOMEOSTATIC_MECHANISMS_KERNEL_CUH
#define HOMEOSTATIC_MECHANISMS_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * Synaptic scaling kernel implementing multiplicative homeostatic plasticity
 * Maintains overall neural activity levels within target ranges
 */
__global__ void synapticScalingKernel(GPUSynapse* synapses,
                                     GPUNeuronState* neurons,
                                     float current_time,
                                     float dt,
                                     int num_synapses,
                                     int num_neurons) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // ========================================
    // UPDATE NEURON ACTIVITY TRACKING
    // ========================================
    
    // Update running average of firing rate
    float decay = expf(-dt / FIRING_RATE_TAU);
    neuron.average_firing_rate *= decay;
    
    // Add current activity contribution
    float current_contribution = neuron.spiked ? 1.0f : 0.0f;
    neuron.average_firing_rate += current_contribution * (1.0f - decay);
    
    // Convert to Hz (assuming dt is in milliseconds)
    float firing_rate_hz = neuron.average_firing_rate * 1000.0f / FIRING_RATE_TAU;
    
    // ========================================
    // COMPUTE SCALING FACTOR
    // ========================================
    
    float scaling_factor = 1.0f;
    
    // Determine if scaling is needed
    if (firing_rate_hz > TARGET_FIRING_RATE * 1.2f) {
        // Firing rate too high: scale down incoming weights
        float excess_ratio = firing_rate_hz / TARGET_FIRING_RATE;
        scaling_factor = 1.0f - SYNAPTIC_SCALING_RATE * (excess_ratio - 1.0f) * dt;
        
    } else if (firing_rate_hz < TARGET_FIRING_RATE * 0.8f) {
        // Firing rate too low: scale up incoming weights
        float deficit_ratio = TARGET_FIRING_RATE / (firing_rate_hz + EPSILON);
        scaling_factor = 1.0f + SYNAPTIC_SCALING_RATE * (deficit_ratio - 1.0f) * dt;
    }
    
    // Limit scaling to prevent instability
    scaling_factor = fmaxf(0.95f, fminf(1.05f, scaling_factor));
    
    // ========================================
    // APPLY SCALING TO INCOMING SYNAPSES
    // ========================================
    
    // This requires a second pass through synapses
    // Store scaling factor in neuron state for subsequent kernel
    neuron.homeostatic_scaling_factor = scaling_factor;
    
    // Update intrinsic excitability if needed
    if (fabsf(scaling_factor - 1.0f) > 0.01f) {
        // Adjust threshold to compensate for synaptic scaling
        float threshold_adjustment = -0.1f * (scaling_factor - 1.0f);
        neuron.threshold += threshold_adjustment * INTRINSIC_EXCITABILITY_RATE * dt;
        
        // Keep threshold within reasonable bounds
        neuron.threshold = fmaxf(-80.0f, fminf(-40.0f, neuron.threshold));
    }
}

/**
 * Apply synaptic scaling factors computed in previous kernel
 */
__global__ void applySynapticScalingKernel(GPUSynapse* synapses,
                                          const GPUNeuronState* neurons,
                                          int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    int post_idx = synapse.post_neuron_idx;
    if (post_idx < 0) return;
    
    // Apply scaling factor from postsynaptic neuron
    float scaling_factor = neurons[post_idx].homeostatic_scaling_factor;
    
    // Only scale excitatory weights (inhibitory weights scale oppositely)
    if (synapse.weight > 0.0f) {
        synapse.weight *= scaling_factor;
    } else {
        // Inhibitory weights scale in opposite direction to maintain balance
        synapse.weight *= (2.0f - scaling_factor);
    }
    
    // Apply bounds
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
}

/**
 * Weight normalization kernel enforcing total weight constraints
 */
__global__ void weightNormalizationKernel(GPUSynapse* synapses,
                                         int* neuron_synapse_counts,
                                         int num_synapses,
                                         int num_neurons) {
    
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    // ========================================
    // COMPUTE TOTAL INCOMING WEIGHTS
    // ========================================
    
    float total_in_weight = 0.0f;
    float total_in_excitatory = 0.0f;
    float total_in_inhibitory = 0.0f;
    int in_count = 0;
    
    for (int s = 0; s < num_synapses; s++) {
        if (synapses[s].active == 0) continue;
        
        if (synapses[s].post_neuron_idx == neuron_idx) {
            float weight = synapses[s].weight;
            total_in_weight += fabsf(weight);
            
            if (weight > 0.0f) {
                total_in_excitatory += weight;
            } else {
                total_in_inhibitory += fabsf(weight);
            }
            in_count++;
        }
    }
    
    // ========================================
    // COMPUTE TOTAL OUTGOING WEIGHTS
    // ========================================
    
    float total_out_weight = 0.0f;
    float total_out_excitatory = 0.0f;
    float total_out_inhibitory = 0.0f;
    int out_count = 0;
    
    for (int s = 0; s < num_synapses; s++) {
        if (synapses[s].active == 0) continue;
        
        if (synapses[s].pre_neuron_idx == neuron_idx) {
            float weight = synapses[s].weight;
            total_out_weight += fabsf(weight);
            
            if (weight > 0.0f) {
                total_out_excitatory += weight;
            } else {
                total_out_inhibitory += fabsf(weight);
            }
            out_count++;
        }
    }
    
    // ========================================
    // NORMALIZE INCOMING WEIGHTS IF NEEDED
    // ========================================
    
    if (total_in_weight > MAX_TOTAL_IN_WEIGHT && in_count > 0) {
        float in_scale_factor = MAX_TOTAL_IN_WEIGHT / total_in_weight;
        
        // Apply scaling to incoming synapses
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].post_neuron_idx == neuron_idx) {
                synapses[s].weight *= in_scale_factor;
            }
        }
    }
    
    // ========================================
    // NORMALIZE OUTGOING WEIGHTS IF NEEDED
    // ========================================
    
    if (total_out_weight > MAX_TOTAL_OUT_WEIGHT && out_count > 0) {
        float out_scale_factor = MAX_TOTAL_OUT_WEIGHT / total_out_weight;
        
        // Apply scaling to outgoing synapses
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].pre_neuron_idx == neuron_idx) {
                synapses[s].weight *= out_scale_factor;
            }
        }
    }
    
    // ========================================
    // MAINTAIN EXCITATION-INHIBITION BALANCE
    // ========================================
    
    // Ideal E/I ratio is approximately 4:1 for cortical circuits
    float ideal_ei_ratio = 4.0f;
    float current_ei_ratio = (total_in_inhibitory > 0.0f) ? 
                            total_in_excitatory / total_in_inhibitory : 10.0f;
    
    if (current_ei_ratio > ideal_ei_ratio * 1.5f) {
        // Too much excitation: slightly strengthen inhibition
        float adjustment_factor = 1.02f;
        
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].post_neuron_idx == neuron_idx && synapses[s].weight < 0.0f) {
                synapses[s].weight *= adjustment_factor;
                if (synapses[s].weight < MIN_WEIGHT) {
                    synapses[s].weight = MIN_WEIGHT;
                }
            }
        }
        
    } else if (current_ei_ratio < ideal_ei_ratio * 0.5f) {
        // Too much inhibition: slightly strengthen excitation
        float adjustment_factor = 1.02f;
        
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].post_neuron_idx == neuron_idx && synapses[s].weight > 0.0f) {
                synapses[s].weight *= adjustment_factor;
                if (synapses[s].weight > MAX_WEIGHT) {
                    synapses[s].weight = MAX_WEIGHT;
                }
            }
        }
    }
}

/**
 * Activity regulation kernel maintaining target activity levels
 */
__global__ void activityRegulationKernel(GPUNeuronState* neurons,
                                        float current_time,
                                        float dt,
                                        int num_neurons) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // ========================================
    // UPDATE ACTIVITY METRICS
    // ========================================
    
    // Update sliding window activity level
    float decay = expf(-dt / ACTIVITY_TAU);
    neuron.average_activity *= decay;
    
    // Add current activity contribution
    neuron.average_activity += neuron.activity_level * (1.0f - decay);
    
    // ========================================
    // INTRINSIC EXCITABILITY ADJUSTMENT
    // ========================================
    
    float activity_error = neuron.average_activity - TARGET_ACTIVITY_LEVEL;
    
    if (fabsf(activity_error) > 0.01f) {
        // Adjust intrinsic excitability to maintain target activity
        float excitability_change = -activity_error * INTRINSIC_EXCITABILITY_RATE * dt;
        
        // Modify threshold (lower threshold = higher excitability)
        neuron.threshold += excitability_change;
        
        // Bounds on threshold adjustment
        neuron.threshold = fmaxf(-80.0f, fminf(-40.0f, neuron.threshold));
        
        // Also adjust leak conductance slightly
        neuron.leak_conductance += excitability_change * 0.1f;
        neuron.leak_conductance = fmaxf(0.1f, fminf(1.0f, neuron.leak_conductance));
    }
    
    // ========================================
    // SPIKE FREQUENCY ADAPTATION
    // ========================================
    
    // Implement activity-dependent spike frequency adaptation
    if (neuron.spiked) {
        // Increase adaptation current after each spike
        neuron.adaptation_current += 0.1f;
    }
    
    // Decay adaptation current
    neuron.adaptation_current *= expf(-dt / 1000.0f);  // 1-second time constant
    
    // Clamp adaptation current
    neuron.adaptation_current = fmaxf(0.0f, fminf(1.0f, neuron.adaptation_current));
    
    // ========================================
    // HOMEOSTATIC PARAMETER UPDATES
    // ========================================
    
    // Update homeostatic time constants based on recent activity
    if (neuron.average_activity > TARGET_ACTIVITY_LEVEL * 1.5f) {
        // High activity: speed up homeostatic mechanisms
        neuron.homeostatic_time_constant *= 0.99f;
    } else if (neuron.average_activity < TARGET_ACTIVITY_LEVEL * 0.5f) {
        // Low activity: slow down homeostatic mechanisms
        neuron.homeostatic_time_constant *= 1.01f;
    }
    
    // Keep time constant within reasonable bounds
    neuron.homeostatic_time_constant = fmaxf(1000.0f, fminf(100000.0f, 
                                           neuron.homeostatic_time_constant));
}

/**
 * Network-wide homeostatic monitoring kernel
 * Computes global network statistics and adjusts parameters accordingly
 */
__global__ void networkHomeostaticMonitoringKernel(const GPUNeuronState* neurons,
                                                  const GPUSynapse* synapses,
                                                  float* network_stats,
                                                  int num_neurons,
                                                  int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for efficient reduction
    __shared__ float activity_sum[256];
    __shared__ float weight_sum[256];
    __shared__ float firing_rate_sum[256];
    __shared__ int active_neuron_count[256];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    activity_sum[tid] = 0.0f;
    weight_sum[tid] = 0.0f;
    firing_rate_sum[tid] = 0.0f;
    active_neuron_count[tid] = 0;
    
    // ========================================
    // COMPUTE LOCAL STATISTICS
    // ========================================
    
    if (idx < num_neurons) {
        const GPUNeuronState& neuron = neurons[idx];
        
        activity_sum[tid] = neuron.average_activity;
        firing_rate_sum[tid] = neuron.average_firing_rate;
        active_neuron_count[tid] = (neuron.average_activity > 0.01f) ? 1 : 0;
    }
    
    if (idx < num_synapses) {
        const GPUSynapse& synapse = synapses[idx];
        
        if (synapse.active) {
            weight_sum[tid] = fabsf(synapse.weight);
        }
    }
    
    __syncthreads();
    
    // ========================================
    // BLOCK-WISE REDUCTION
    // ========================================
    
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            activity_sum[tid] += activity_sum[tid + stride];
            weight_sum[tid] += weight_sum[tid + stride];
            firing_rate_sum[tid] += firing_rate_sum[tid + stride];
            active_neuron_count[tid] += active_neuron_count[tid + stride];
        }
        __syncthreads();
    }
    
    // ========================================
    // WRITE TO GLOBAL MEMORY
    // ========================================
    
    if (tid == 0) {
        atomicAdd(&network_stats[0], activity_sum[0]);      // Total network activity
        atomicAdd(&network_stats[1], weight_sum[0]);        // Total synaptic weight
        atomicAdd(&network_stats[2], firing_rate_sum[0]);   // Total firing rate
        atomicAdd((int*)&network_stats[3], active_neuron_count[0]); // Active neuron count
    }
}

/**
 * Emergency stabilization kernel for pathological network states
 */
__global__ void emergencyStabilizationKernel(GPUSynapse* synapses,
                                            GPUNeuronState* neurons,
                                            float network_activity_level,
                                            float emergency_threshold,
                                            int num_synapses,
                                            int num_neurons) {
    
    // Check if emergency intervention is needed
    if (network_activity_level < emergency_threshold * 0.1f || 
        network_activity_level > emergency_threshold * 10.0f) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Stabilize synapses
        if (idx < num_synapses) {
            GPUSynapse& synapse = synapses[idx];
            
            if (synapse.active) {
                // Reset to moderate values
                if (fabsf(synapse.weight) > MAX_WEIGHT * 0.8f) {
                    synapse.weight *= 0.8f;
                }
                
                // Reset eligibility traces to prevent runaway plasticity
                synapse.fast_trace *= 0.5f;
                synapse.medium_trace *= 0.7f;
                synapse.slow_trace *= 0.9f;
                synapse.tag_strength *= 0.5f;
            }
        }
        
        // Stabilize neurons
        if (idx < num_neurons) {
            GPUNeuronState& neuron = neurons[idx];
            
            // Reset extreme activity levels
            if (neuron.average_activity > 1.0f) {
                neuron.average_activity = 0.5f;
            } else if (neuron.average_activity < 0.001f) {
                neuron.average_activity = 0.01f;
            }
            
            // Reset threshold to default
            neuron.threshold = -55.0f;
            
            // Clear adaptation current
            neuron.adaptation_current *= 0.1f;
        }
    }
}

#endif // HOMEOSTATIC_MECHANISMS_KERNEL_CUH