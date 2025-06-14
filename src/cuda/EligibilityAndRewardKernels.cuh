#ifndef ELIGIBILITY_AND_REWARD_KERNELS_CUH
#define ELIGIBILITY_AND_REWARD_KERNELS_CUH

#include <NeuroGen/cuda/GPUNeuralStructures.h>

/**
 * @brief Resets the eligibility traces for synapses.
 *
 * This kernel can reset positive, negative, or all eligibility traces, which is
 * crucial for starting new learning episodes or managing trace decay.
 */
__global__ void eligibilityTraceResetKernel(GPUSynapse* synapses, int num_synapses, bool reset_positive, bool reset_negative, bool reset_all);

/**
 * @brief Monitors the state of eligibility traces for debugging and analysis.
 *
 * This kernel extracts the values of eligibility traces and stores them in a
 * buffer for analysis on the host.
 */
__global__ void traceMonitoringKernel(const GPUSynapse* synapses, int num_synapses, float* trace_buffer);

/**
 * @brief Adapts the sensitivity of synapses to dopamine based on neural activity.
 *
 * This mechanism allows the network to dynamically adjust how it responds to
 * reward signals, a key feature of advanced reinforcement learning.
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float base_sensitivity, float max_sensitivity, float adaptation_rate, int num_synapses);

/**
 * @brief Updates the global reward trace based on external reward signals.
 *
 * The reward trace integrates rewards over time, providing a smoother signal
 * for learning and credit assignment.
 */
__global__ void rewardTraceUpdateKernel(float* reward_trace, float current_reward, float decay_rate);

/**
 * @brief Updates the eligibility traces based on pre- and post-synaptic activity.
 *
 * This kernel implements the core mechanism of trace-based learning rules like
 * STDP with eligibility traces.
 */
__global__ void eligibilityTraceUpdateKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float tau_e, float a_pre, int num_synapses);

/**
 * @brief Implements the late-phase (long-term) plasticity consolidation.
 *
 * This kernel strengthens synapses that have high eligibility traces, leading
 * to long-term memory formation.
 */
__global__ void latePhaseePlasticityKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float learning_rate, float eligibility_threshold, float consolidation_factor, int num_synapses);

/**
 * @brief Calculates the reward prediction error (TD error).
 *
 * This is a fundamental computation in reinforcement learning, where the network
 * learns to predict future rewards.
 */
__global__ void rewardPredictionErrorKernel(const GPUNeuronState* neurons, float actual_reward, float* value_estimate, float* last_value_estimate, float* td_error, float discount_factor, float learning_rate, int num_neurons);

/**
 * @brief Modulates synaptic weights based on a reward signal and eligibility traces.
 *
 * This kernel applies the reward signal to eligible synapses, driving the
 * learning process in the direction of rewarding outcomes.
 */
__global__ void rewardModulationKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, float learning_rate, float reward_signal, float eligibility_trace_decay, float baseline_reward, float modulation_factor, int num_synapses);

#endif // ELIGIBILITY_AND_REWARD_KERNELS_CUH