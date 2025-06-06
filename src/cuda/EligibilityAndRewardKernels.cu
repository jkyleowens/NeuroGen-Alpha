#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/LearningRuleConstants.hh>

/**
 * @brief Manages the passive dynamics of eligibility traces.
 *
 * This kernel applies exponential decay to all traces and handles the "cascade"
 * where fast traces consolidate into medium traces, and medium into slow,
 * mimicking memory consolidation.
 */
__global__ void eligibilityTraceUpdateKernel(GPUSynapse* synapses, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // 1. Decay all traces over time
    float fast_decay = expf(-dt / FAST_TRACE_TAU);
    float medium_decay = expf(-dt / MEDIUM_TRACE_TAU);
    float slow_decay = expf(-dt / SLOW_TRACE_TAU);

    synapse.eligibility_trace *= fast_decay;
    synapse.medium_trace *= medium_decay;
    synapse.slow_trace *= slow_decay;

    // 2. Cascade from fast to medium trace
    float fast_to_medium = synapse.eligibility_trace * FAST_TO_MEDIUM_TRANSFER * dt;
    synapse.medium_trace += fast_to_medium;

    // 3. Cascade from medium to slow trace
    float medium_to_slow = synapse.medium_trace * MEDIUM_TO_SLOW_TRANSFER * dt;
    synapse.slow_trace += medium_to_slow;

    // 4. Clamp traces to prevent runaway values
    synapse.eligibility_trace = fmaxf(-MAX_ELIGIBILITY_TRACE, fminf(MAX_ELIGIBILITY_TRACE, synapse.eligibility_trace));
    synapse.medium_trace = fmaxf(-MAX_ELIGIBILITY_TRACE, fminf(MAX_ELIGIBILITY_TRACE, synapse.medium_trace));
    synapse.slow_trace = fmaxf(-MAX_ELIGIBILITY_TRACE, fminf(MAX_ELIGIBILITY_TRACE, synapse.slow_trace));
}


/**
 * @brief Applies a global reward signal to consolidate learning.
 *
 * This kernel models the effect of neuromodulators like dopamine. It uses the
 * reward signal to convert the eligibility traces into permanent weight changes.
 * Stronger traces (both positive and negative) are affected more by the reward.
 */
__global__ void rewardModulationKernel(GPUSynapse* synapses, float reward_signal, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // Reward signal only affects traces that have been "tagged" by recent activity
    if (fabsf(synapse.eligibility_trace) < 0.001f && fabsf(synapse.medium_trace) < 0.001f) {
        return;
    }

    // Combine traces to determine total plasticity potential
    float plasticity_potential = synapse.eligibility_trace * FAST_TRACE_MODULATION +
                                 synapse.medium_trace * MEDIUM_TRACE_MODULATION +
                                 synapse.slow_trace * SLOW_TRACE_MODULATION;

    // The reward signal gates the consolidation of this potential into a weight change
    float weight_change = HEBBIAN_LEARNING_RATE * reward_signal * plasticity_potential;

    if (fabsf(weight_change) > 0.0f) {
        atomicAdd(&synapse.weight, weight_change);
        
        // After consolidation, consume the traces to reset them for future learning
        atomicExch(&synapse.eligibility_trace, synapse.eligibility_trace * 0.1f);
        atomicExch(&synapse.medium_trace, synapse.medium_trace * 0.2f);
    }
    
    // Clamp final weight to its bounds
    synapse.weight = fmaxf(MIN_WEIGHT, fminf(MAX_WEIGHT, synapse.weight));
}