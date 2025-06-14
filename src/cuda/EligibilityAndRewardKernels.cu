// ============================================================================
// EligibilityAndRewardKernels.cu - IMPLEMENTATIONS
// ============================================================================

#include <NeuroGen/cuda/EligibilityAndRewardKernels.cuh>
#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Define missing constants
#ifndef REFRACTORY_PERIOD_MS
#define REFRACTORY_PERIOD_MS 2.0f  // 2ms refractory period
#endif

/**
 * Reset eligibility traces in synapses
 */
__global__ void eligibilityTraceResetKernel(GPUSynapse* synapses, 
                                           int num_synapses, 
                                           bool reset_all,
                                           bool reset_positive_only, 
                                           bool reset_negative_only) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (reset_all) {
        synapse.eligibility_trace = 0.0f;
    } else if (reset_positive_only && synapse.eligibility_trace > 0.0f) {
        synapse.eligibility_trace = 0.0f;
    } else if (reset_negative_only && synapse.eligibility_trace < 0.0f) {
        synapse.eligibility_trace = 0.0f;
    }
}

/**
 * Monitor and collect eligibility trace statistics
 */
__global__ void traceMonitoringKernel(const GPUSynapse* synapses, 
                                     int num_synapses, 
                                     float* trace_stats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    const GPUSynapse& synapse = synapses[idx];
    
    // Store trace value for this synapse (could be used for analysis)
    if (trace_stats) {
        trace_stats[idx] = synapse.eligibility_trace;
    }
    
    // Could add more sophisticated monitoring here
    // e.g., compute running averages, detect anomalies, etc.
}

/**
 * Adapt dopamine sensitivity based on neuron activity
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   const GPUNeuronState* neurons,
                                                   float adaptation_rate,
                                                   float target_activity,
                                                   float current_dopamine,
                                                   int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Get post-synaptic neuron activity
    int post_neuron_idx = synapse.post_neuron_idx;
    if (post_neuron_idx >= 0) {
        const GPUNeuronState& post_neuron = neurons[post_neuron_idx];
        
        // Calculate activity metric (simple spike-based for now)
        float activity = post_neuron.spiked ? 1.0f : 0.0f;
        
        // Adapt dopamine sensitivity based on activity difference
        float activity_error = activity - target_activity;
        
        // Adjust dopamine sensitivity (stored in a synapse field)
        // This is a simplified model - could be more sophisticated
        synapse.dopamine_sensitivity += adaptation_rate * activity_error * current_dopamine;
        
        // Clamp sensitivity to reasonable bounds
        synapse.dopamine_sensitivity = fmaxf(0.1f, fminf(2.0f, synapse.dopamine_sensitivity));
    }
}

/**
 * Update reward traces for reinforcement learning
 */
__global__ void rewardTraceUpdateKernel(float* reward_traces,
                                       float decay_factor,
                                       float current_reward) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update reward trace with decay and current reward
    reward_traces[idx] = reward_traces[idx] * decay_factor + current_reward;
}

// ============================================================================
// WRAPPER FUNCTION IMPLEMENTATIONS
// ============================================================================

void launchEligibilityTraceReset(GPUSynapse* d_synapses, 
                                int num_synapses,
                                bool reset_all,
                                bool reset_positive_only,
                                bool reset_negative_only) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    eligibilityTraceResetKernel<<<grid, block>>>(d_synapses, num_synapses,
                                                 reset_all, reset_positive_only, 
                                                 reset_negative_only);
    cudaDeviceSynchronize();
}

void launchTraceMonitoring(const GPUSynapse* d_synapses,
                          int num_synapses,
                          float* d_trace_stats) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    traceMonitoringKernel<<<grid, block>>>(d_synapses, num_synapses, d_trace_stats);
    cudaDeviceSynchronize();
}

void launchDopamineSensitivityAdaptation(GPUSynapse* d_synapses,
                                        const GPUNeuronState* d_neurons,
                                        int num_synapses,
                                        float adaptation_rate,
                                        float target_activity,
                                        float current_dopamine) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    dopamineSensitivityAdaptationKernel<<<grid, block>>>(d_synapses, d_neurons,
                                                         adaptation_rate, target_activity,
                                                         current_dopamine, num_synapses);
    cudaDeviceSynchronize();
}

void launchRewardTraceUpdate(float* d_reward_traces,
                            int num_traces,
                            float decay_factor,
                            float current_reward) {
    dim3 block(256);
    dim3 grid((num_traces + block.x - 1) / block.x);
    
    rewardTraceUpdateKernel<<<grid, block>>>(d_reward_traces, decay_factor, current_reward);
    cudaDeviceSynchronize();
}

// ============================================================================
// NeuronSpikingKernels.cu - IMPLEMENTATIONS
// ============================================================================

// ============================================================================
// WRAPPER FUNCTION IMPLEMENTATIONS FOR ELIGIBILITY AND REWARD KERNELS ONLY
// ============================================================================