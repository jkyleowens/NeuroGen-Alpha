#include <NeuroGen/cuda/EligibilityTraceKernel.cuh>
#include <NeuroGen/cuda/RewardModulationKernel.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/LearningRuleConstants.h>
#include <math.h>

/**
 * @file EligibilityAndRewardKernels.cu
 * @brief Implements the CUDA kernels for multi-timescale eligibility traces,
 * synaptic tagging, reward prediction error, and reward-modulated plasticity.
 */

// --- Eligibility Trace Kernels ---

__global__ void eligibilityTraceUpdateKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float current_time, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0 || !synapse.is_plastic) return;

    // 1. Decay all traces using time constants
    synapse.fast_trace *= expf(-dt / FAST_TRACE_TAU);
    synapse.medium_trace *= expf(-dt / MEDIUM_TRACE_TAU);
    synapse.slow_trace *= expf(-dt / SLOW_TRACE_TAU);
    synapse.tag_strength *= expf(-dt / TAG_TAU);

    // 2. Cascade from fast to medium trace (consolidation)
    float fast_to_medium_transfer = synapse.fast_trace * FAST_TO_MEDIUM_RATE * dt;
    synapse.medium_trace += fast_to_medium_transfer;

    // 3. Cascade from medium to slow trace (long-term consolidation)
    float medium_to_slow_transfer = synapse.medium_trace * MEDIUM_TO_SLOW_RATE * dt;
    synapse.slow_trace += medium_to_slow_transfer;
    
    // 4. Synaptic Tagging: Mark synapses for late-phase plasticity if medium trace is high
    if (fabsf(synapse.medium_trace) > TAG_THRESHOLD) {
        synapse.tag_strength += synapse.medium_trace * TAG_CREATION_RATE * dt;
    }

    // 5. Clamp all traces to prevent runaway values
    synapse.fast_trace = fmaxf(-MAX_FAST_TRACE, fminf(MAX_FAST_TRACE, synapse.fast_trace));
    synapse.medium_trace = fmaxf(-MAX_MEDIUM_TRACE, fminf(MAX_MEDIUM_TRACE, synapse.medium_trace));
    synapse.slow_trace = fmaxf(-MAX_SLOW_TRACE, fminf(MAX_SLOW_TRACE, synapse.slow_trace));
    synapse.tag_strength = fmaxf(-MAX_TAG_STRENGTH, fminf(MAX_TAG_STRENGTH, synapse.tag_strength));
}

__global__ void latePhaseePlasticityKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float protein_synthesis_signal, float current_time, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0 || !synapse.is_plastic) return;

    // Late-phase plasticity requires a strong tag AND a global protein synthesis signal
    if (fabsf(synapse.tag_strength) < PROTEIN_SYNTHESIS_THRESHOLD || protein_synthesis_signal < PROTEIN_SYNTHESIS_THRESHOLD) {
        return;
    }

    // The weight change is proportional to the tag strength and the signal
    float late_phase_dw = synapse.tag_strength * protein_synthesis_signal * LATE_PHASE_FACTOR * dt;
    synapse.weight += late_phase_dw;

    // Consume the tag to prevent repeated, uncontrolled potentiation/depression
    synapse.tag_strength *= 0.3f; 

    // Clamp weight to its bounds
    synapse.weight = fmaxf(MIN_WEIGHT, fminf(MAX_WEIGHT, synapse.weight));
}

// --- Reward Modulation Kernels ---

__global__ void rewardPredictionErrorKernel(const GPUNeuronState* neurons, float actual_reward, float* predicted_reward, float* prediction_error, float* dopamine_level, float current_time, float dt, int num_neurons) {
    // This kernel is intended to be run with a single thread.
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;

    // --- Compute Reward Prediction (V(s)) ---
    // (Simplified: In a full implementation, this would come from specific reward-predicting neurons)
    // For now, we use a placeholder logic.
    *predicted_reward = *dopamine_level; // A simple assumption that last dopamine level is the prediction.

    // --- Compute Prediction Error (δ) ---
    // δ = r - V(s)
    *prediction_error = (actual_reward - *predicted_reward) * PREDICTION_ERROR_SCALE;

    // --- Update Dopamine Level ---
    // The global dopamine level decays and is phasically updated by the prediction error.
    *dopamine_level *= expf(-dt / DOPAMINE_TAU);
    *dopamine_level += *prediction_error;
}

__global__ void rewardModulationKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float reward_signal, float dopamine_level, float prediction_error, float current_time, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0 || !synapse.is_plastic) return;

    // The effective reward signal is the global dopamine level
    float effective_reward = dopamine_level;

    // Modulate each trace by its sensitivity to dopamine
    float modulated_fast = synapse.fast_trace * effective_reward * FAST_TRACE_DOPAMINE_SENS;
    float modulated_medium = synapse.medium_trace * effective_reward * MEDIUM_TRACE_DOPAMINE_SENS;
    float modulated_slow = synapse.slow_trace * effective_reward * SLOW_TRACE_DOPAMINE_SENS;
    
    // The total weight change is the sum of the modulated traces
    float total_dw = (modulated_fast + modulated_medium + modulated_slow) * synapse.plasticity_rate * dt;
    
    synapse.weight += total_dw;
    
    // Clamp the weight to its min/max bounds
    synapse.weight = fmaxf(synapse.min_weight, fminf(synapse.max_weight, synapse.weight));
}