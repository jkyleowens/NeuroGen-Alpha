#ifndef ELIGIBILITY_TRACE_KERNEL_CUH
#define ELIGIBILITY_TRACE_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

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