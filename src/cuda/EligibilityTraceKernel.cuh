#ifndef ELIGIBILITY_TRACE_KERNEL_CUH
#define ELIGIBILITY_TRACE_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * Multi-timescale eligibility trace update kernel
 * Implements the biological synaptic tagging and capture mechanism
 * with fast, medium, and slow eligibility traces
 */
__global__ void eligibilityTraceUpdateKernel(GPUSynapse* synapses, 
                                            const GPUNeuronState* neurons,
                                            float current_time, 
                                            float dt, 
                                            int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    // ========================================
    // DECAY ALL ELIGIBILITY TRACES
    // ========================================
    
    // Compute decay factors for different timescales
    float fast_decay = expf(-dt / FAST_TRACE_TAU);
    float medium_decay = expf(-dt / MEDIUM_TRACE_TAU);
    float slow_decay = expf(-dt / SLOW_TRACE_TAU);
    float tag_decay = expf(-dt / TAG_TAU);
    
    // Apply exponential decay to all traces
    synapse.fast_trace *= fast_decay;
    synapse.medium_trace *= medium_decay;
    synapse.slow_trace *= slow_decay;
    synapse.tag_strength *= tag_decay;
    
    // ========================================
    // TRACE CASCADE MECHANISM
    // ========================================
    
    // Transfer from fast to medium trace (early-phase to intermediate)
    // This represents the consolidation from immediate plasticity to
    // short-term memory formation
    float fast_to_medium_transfer = synapse.fast_trace * FAST_TO_MEDIUM_RATE * dt;
    synapse.medium_trace += fast_to_medium_transfer;
    
    // Apply saturation to prevent unbounded growth
    if (fabsf(synapse.medium_trace) > MAX_MEDIUM_TRACE) {
        synapse.medium_trace = (synapse.medium_trace > 0.0f) ? 
                               MAX_MEDIUM_TRACE : -MAX_MEDIUM_TRACE;
    }
    
    // Transfer from medium to slow trace (intermediate to long-term)
    // This represents the consolidation to long-term memory
    float medium_to_slow_transfer = synapse.medium_trace * MEDIUM_TO_SLOW_RATE * dt;
    synapse.slow_trace += medium_to_slow_transfer;
    
    // Apply saturation to slow trace
    if (fabsf(synapse.slow_trace) > MAX_SLOW_TRACE) {
        synapse.slow_trace = (synapse.slow_trace > 0.0f) ? 
                             MAX_SLOW_TRACE : -MAX_SLOW_TRACE;
    }
    
    // ========================================
    // SYNAPTIC TAGGING MECHANISM
    // ========================================
    
    // Create or strengthen synaptic tags based on medium trace activity
    // Tags mark synapses for potential late-phase plasticity
    if (fabsf(synapse.medium_trace) > TAG_THRESHOLD) {
        
        // Tag strength increases proportional to medium trace magnitude
        float tag_increment = synapse.medium_trace * TAG_CREATION_RATE * dt;
        synapse.tag_strength += tag_increment;
        
        // Apply tag strength constraints
        if (synapse.tag_strength > MAX_TAG_STRENGTH) {
            synapse.tag_strength = MAX_TAG_STRENGTH;
        } else if (synapse.tag_strength < -MAX_TAG_STRENGTH) {
            synapse.tag_strength = -MAX_TAG_STRENGTH;
        }
    }
    
    // ========================================
    // TRACE INTERACTION AND NORMALIZATION
    // ========================================
    
    // Implement trace competition - stronger traces can suppress weaker ones
    // This prevents runaway trace accumulation
    float total_trace_magnitude = fabsf(synapse.fast_trace) + 
                                 fabsf(synapse.medium_trace) + 
                                 fabsf(synapse.slow_trace);
    
    if (total_trace_magnitude > MAX_MEDIUM_TRACE * 2.0f) {
        float normalization_factor = (MAX_MEDIUM_TRACE * 2.0f) / total_trace_magnitude;
        synapse.fast_trace *= normalization_factor;
        synapse.medium_trace *= normalization_factor;
        synapse.slow_trace *= normalization_factor;
    }
    
    // ========================================
    // ACTIVITY-DEPENDENT TRACE MODULATION
    // ========================================
    
    // Get neuron activity levels
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx >= 0 && post_idx >= 0) {
        float pre_activity = neurons[pre_idx].activity_level;
        float post_activity = neurons[post_idx].activity_level;
        
        // Traces are enhanced by correlated pre/post activity
        float activity_correlation = pre_activity * post_activity;
        
        if (activity_correlation > CORRELATION_THRESHOLD) {
            // Boost all traces when neurons are co-active
            float boost_factor = 1.0f + activity_correlation * ACTIVITY_SCALING_FACTOR;
            synapse.fast_trace *= boost_factor;
            synapse.medium_trace *= boost_factor;
            // Don't boost slow trace as much (it represents consolidated memory)
            synapse.slow_trace *= sqrtf(boost_factor);
        }
    }
    
    // ========================================
    // COMPARTMENT-SPECIFIC TRACE DYNAMICS
    // ========================================
    
    // Different compartments have different trace dynamics
    int compartment = synapse.post_compartment;
    int compartment_type = (compartment == 0) ? COMPARTMENT_SOMA : 
                          (compartment <= 3) ? COMPARTMENT_BASAL : COMPARTMENT_APICAL;
    
    switch (compartment_type) {
        case COMPARTMENT_APICAL:
            // Apical dendrites have enhanced medium-term traces
            // This reflects their role in top-down processing and context
            synapse.medium_trace *= 1.1f;
            break;
            
        case COMPARTMENT_BASAL:
            // Basal dendrites have enhanced fast traces
            // This reflects their role in bottom-up processing
            synapse.fast_trace *= 1.05f;
            break;
            
        case COMPARTMENT_SOMA:
            // Somatic synapses have balanced trace dynamics
            // No modification needed
            break;
    }
    
    // ========================================
    // TRACE SATURATION AND BOUNDS
    // ========================================
    
    // Apply final bounds to prevent numerical issues
    synapse.fast_trace = fmaxf(-MAX_FAST_TRACE, fminf(MAX_FAST_TRACE, synapse.fast_trace));
    synapse.medium_trace = fmaxf(-MAX_MEDIUM_TRACE, fminf(MAX_MEDIUM_TRACE, synapse.medium_trace));
    synapse.slow_trace = fmaxf(-MAX_SLOW_TRACE, fminf(MAX_SLOW_TRACE, synapse.slow_trace));
}

/**
 * Late-phase plasticity kernel implementing protein synthesis-dependent 
 * long-term potentiation and depression
 */
__global__ void latePhaseePlasticityKernel(GPUSynapse* synapses, 
                                          const GPUNeuronState* neurons,
                                          float protein_synthesis_signal,
                                          float current_time,
                                          float dt,
                                          int num_synapses) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    // ========================================
    // PROTEIN SYNTHESIS THRESHOLD CHECK
    // ========================================
    
    // Late-phase plasticity requires both a strong tag and protein synthesis signal
    if (fabsf(synapse.tag_strength) < PROTEIN_SYNTHESIS_THRESHOLD || 
        protein_synthesis_signal < PROTEIN_SYNTHESIS_THRESHOLD) {
        return;
    }
    
    // ========================================
    // COMPUTE LATE-PHASE WEIGHT CHANGE
    // ========================================
    
    // The direction of plasticity is determined by the tag sign
    // The magnitude is determined by both tag strength and protein synthesis
    float late_phase_dw = synapse.tag_strength * 
                         protein_synthesis_signal * 
                         LATE_PHASE_FACTOR * dt;
    
    // ========================================
    // APPLY LATE-PHASE PLASTICITY
    // ========================================
    
    // Late-phase changes are typically much larger than early-phase
    synapse.weight += late_phase_dw;
    
    // Apply constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Maintain synapse type consistency
    bool is_excitatory = synapse.weight > 0.0f;
    if (is_excitatory && synapse.weight < 0.0f) {
        synapse.weight = 0.0f;
    } else if (!is_excitatory && synapse.weight > 0.0f) {
        synapse.weight = 0.0f;
    }
    
    // ========================================
    // TAG CONSUMPTION
    // ========================================
    
    // Successful late-phase plasticity consumes the synaptic tag
    // This prevents repeated triggering and implements tag competition
    synapse.tag_strength *= 0.3f;  // Partial consumption
    
    // Update slow trace to reflect successful consolidation
    synapse.slow_trace += late_phase_dw * 0.5f;
    
    // Clamp slow trace
    if (fabsf(synapse.slow_trace) > MAX_SLOW_TRACE) {
        synapse.slow_trace = (synapse.slow_trace > 0.0f) ? 
                             MAX_SLOW_TRACE : -MAX_SLOW_TRACE;
    }
}

/**
 * Eligibility trace reset kernel for episodic learning
 * Resets traces at episode boundaries while preserving long-term memory
 */
__global__ void eligibilityTraceResetKernel(GPUSynapse* synapses, 
                                           int num_synapses,
                                           bool reset_fast,
                                           bool reset_medium,
                                           bool reset_slow) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    // Selective trace reset based on learning paradigm
    if (reset_fast) {
        synapse.fast_trace = 0.0f;
    }
    
    if (reset_medium) {
        synapse.medium_trace *= 0.1f;  // Partial reset to preserve some memory
    }
    
    if (reset_slow) {
        synapse.slow_trace *= 0.5f;   // Conservative reset for long-term memory
    }
    
    // Tags are typically preserved across episodes
    // They represent the capacity for future learning
}

/**
 * Trace monitoring kernel for debugging and analysis
 * Computes network-wide trace statistics
 */
__global__ void traceMonitoringKernel(const GPUSynapse* synapses,
                                     int num_synapses,
                                     float* trace_stats) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    const GPUSynapse& synapse = synapses[idx];
    
    if (synapse.active == 0) return;
    
    // Compute per-thread statistics
    __shared__ float fast_sum[256];
    __shared__ float medium_sum[256];
    __shared__ float slow_sum[256];
    __shared__ float tag_sum[256];
    
    int tid = threadIdx.x;
    
    fast_sum[tid] = fabsf(synapse.fast_trace);
    medium_sum[tid] = fabsf(synapse.medium_trace);
    slow_sum[tid] = fabsf(synapse.slow_trace);
    tag_sum[tid] = fabsf(synapse.tag_strength);
    
    __syncthreads();
    
    // Block-wise reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            fast_sum[tid] += fast_sum[tid + stride];
            medium_sum[tid] += medium_sum[tid + stride];
            slow_sum[tid] += slow_sum[tid + stride];
            tag_sum[tid] += tag_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(&trace_stats[0], fast_sum[0]);     // Total fast trace activity
        atomicAdd(&trace_stats[1], medium_sum[0]);   // Total medium trace activity
        atomicAdd(&trace_stats[2], slow_sum[0]);     // Total slow trace activity
        atomicAdd(&trace_stats[3], tag_sum[0]);      // Total tag strength
    }
}

#endif // ELIGIBILITY_TRACE_KERNEL_CUH