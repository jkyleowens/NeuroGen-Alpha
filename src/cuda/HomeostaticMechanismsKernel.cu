#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>
#include <math.h>
#include <stdio.h>

/**
 * @file HomeostaticMechanismsKernel.cu
 * @brief Implements all CUDA kernels for homeostatic plasticity rules to ensure network stability.
 * This includes synaptic scaling, weight normalization, activity regulation, network monitoring,
 * and emergency stabilization procedures.
 */

// ====================================================================================
// KERNEL IMPLEMENTATIONS
// ====================================================================================


/**
 * @brief Updates each neuron's average firing rate and computes a scaling
 * factor to apply to its incoming synapses.
 *
 * This kernel is the first step in synaptic scaling. It calculates how much a
 * neuron's synapses need to be scaled up or down to guide the neuron back to
 * its target firing rate. The computed factor is stored on the neuron.
 */
__global__ void computeSynapticScalingFactorKernel(GPUNeuronState* neurons, float dt, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];
    if (neuron.active == 0) return;

    // 1. Update the running average of the neuron's firing rate.
    // This uses an exponential moving average for efficiency.
    float decay_factor = expf(-dt / FIRING_RATE_TAU);
    float current_spike = neuron.spiked ? 1.0f / dt : 0.0f; // Instantaneous rate
    neuron.avg_firing_rate = neuron.avg_firing_rate * decay_factor + current_spike * (1.0f - decay_factor);

    // 2. Calculate the error between the current rate and the homeostatic target rate.
    float rate_error = neuron.avg_firing_rate - TARGET_FIRING_RATE;

    // 3. Compute the scaling factor. This is a slow, multiplicative adjustment.
    float scaling_adjustment = -rate_error * SYNAPTIC_SCALING_RATE * dt;
    float scaling_factor = 1.0f + scaling_adjustment;

    // Clamp the scaling factor to prevent extreme, destabilizing changes.
    neuron.homeostatic_scaling_factor = fmaxf(0.999f, fminf(1.001f, scaling_factor));
}

/**
 * @brief Applies the computed scaling factor to all synapses targeting each neuron.
 *
 * This kernel must be run after computeSynapticScalingFactorKernel. It iterates
 * through all synapses and adjusts their weights based on the scaling factor
 * stored on the postsynaptic neuron.
 */
__global__ void applySynapticScalingKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, int num_synapses) 
{
    
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
 * @brief Enforces total weight constraints and E/I balance for each neuron.
 * @note This kernel is computationally expensive due to its nested loops. It should be run
 * infrequently (e.g., every few seconds of simulation time) to avoid performance bottlenecks.
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
    
    for (int s = 0; s < num_synapses; s++) {
        if (synapses[s].active == 0) continue;
        
        if (synapses[s].pre_neuron_idx == neuron_idx) {
            total_out_weight += fabsf(synapses[s].weight);
        }
    }
    
    // ========================================
    // NORMALIZE INCOMING WEIGHTS IF NEEDED
    // ========================================
    
    if (total_in_weight > MAX_TOTAL_IN_WEIGHT && total_in_weight > EPSILON) {
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
    
    if (total_out_weight > MAX_TOTAL_OUT_WEIGHT && total_out_weight > EPSILON) {
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
    
    float current_ei_ratio = (total_in_inhibitory > EPSILON) ? 
                            total_in_excitatory / total_in_inhibitory : 10.0f;
    
    if (current_ei_ratio > E_I_RATIO_TARGET * 1.5f) {
        // Too much excitation: slightly strengthen inhibition
        float adjustment_factor = 1.02f;
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].post_neuron_idx == neuron_idx && synapses[s].weight < 0.0f) {
                synapses[s].weight = fmaxf(synapses[s].weight * adjustment_factor, MIN_WEIGHT);
            }
        }
        
    } else if (current_ei_ratio < E_I_RATIO_TARGET * 0.5f) {
        // Too much inhibition: slightly strengthen excitation
        float adjustment_factor = 1.02f;
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].post_neuron_idx == neuron_idx && synapses[s].weight > 0.0f) {
                synapses[s].weight = fminf(synapses[s].weight * adjustment_factor, MAX_WEIGHT);
            }
        }
    }
}

/**
 * @brief Adjusts intrinsic neuron properties to regulate activity levels and spike frequency.
 */
__global__ void activityRegulationKernel(GPUNeuronState* neurons,
                                        float current_time,
                                        float dt,
                                        int num_neurons) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Update sliding window of average activity (a proxy for sub-threshold depolarization)
    float decay = expf(-dt / ACTIVITY_TAU);
    neuron.average_activity = neuron.average_activity * decay + neuron.voltage * (1.0f - decay);
    
    // Intrinsic Plasticity: Adjust threshold based on activity error
    float activity_error = neuron.average_activity - TARGET_ACTIVITY_LEVEL;
    
    if (fabsf(activity_error) > 0.01f) {
        float excitability_change = -activity_error * INTRINSIC_EXCITABILITY_RATE * dt;
        neuron.threshold = fmaxf(-80.0f, fminf(-40.0f, neuron.threshold + excitability_change));
    }
    
    // Spike Frequency Adaptation: Simple model where each spike increases an adaptation current
    if (neuron.spiked) {
        neuron.adaptation_current += ADAPTATION_INCREMENT;
    }
    // Adaptation current decays over time
    neuron.adaptation_current *= expf(-dt / ADAPTATION_TAU); 
}

/**
 * @brief Computes global network statistics for high-level monitoring.
 * @note This kernel uses an efficient parallel reduction in shared memory.
 */
__global__ void networkHomeostaticMonitoringKernel(const GPUNeuronState* neurons,
                                                  const GPUSynapse* synapses,
                                                  float* network_stats, // Output buffer on device
                                                  int num_neurons,
                                                  int num_synapses) {
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Zero out shared memory
    sdata[tid] = 0;
    __syncthreads();

    // Each thread computes a partial sum
    while (i < num_neurons) {
        sdata[tid] += neurons[i].avg_firing_rate;
        i += gridDim.x * blockDim.x;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(&network_stats[0], sdata[0]); // Index 0: Mean Firing Rate (will be divided by num_neurons on host)
    }
}

/**
 * @brief Applies strong, global dampening if network activity becomes pathological (e.g., seizure-like).
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