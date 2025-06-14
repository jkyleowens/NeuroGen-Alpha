// ============================================================================
// EligibilityAndRewardKernels.cuh - DECLARATIONS ONLY
// ============================================================================

#ifndef ELIGIBILITY_AND_REWARD_KERNELS_CUH
#define ELIGIBILITY_AND_REWARD_KERNELS_CUH

#include <cuda_runtime.h>
#include <NeuroGen/GPUNeuralStructures.h>

// KERNEL DECLARATIONS (NO DEFINITIONS IN HEADER)

/**
 * Reset eligibility traces in synapses
 */
__global__ void eligibilityTraceResetKernel(GPUSynapse* synapses, 
                                           int num_synapses, 
                                           bool reset_all,
                                           bool reset_positive_only, 
                                           bool reset_negative_only);

/**
 * Monitor and collect eligibility trace statistics
 */
__global__ void traceMonitoringKernel(const GPUSynapse* synapses, 
                                     int num_synapses, 
                                     float* trace_stats);

/**
 * Adapt dopamine sensitivity based on neuron activity
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   const GPUNeuronState* neurons,
                                                   float adaptation_rate,
                                                   float target_activity,
                                                   float current_dopamine,
                                                   int num_synapses);

/**
 * Update reward traces for reinforcement learning
 */
__global__ void rewardTraceUpdateKernel(float* reward_traces,
                                       float decay_factor,
                                       float current_reward);

// WRAPPER FUNCTION DECLARATIONS

void launchEligibilityTraceReset(GPUSynapse* d_synapses, 
                                int num_synapses,
                                bool reset_all = true,
                                bool reset_positive_only = false,
                                bool reset_negative_only = false);

void launchTraceMonitoring(const GPUSynapse* d_synapses,
                          int num_synapses,
                          float* d_trace_stats);

void launchDopamineSensitivityAdaptation(GPUSynapse* d_synapses,
                                        const GPUNeuronState* d_neurons,
                                        int num_synapses,
                                        float adaptation_rate = 0.001f,
                                        float target_activity = 0.1f,
                                        float current_dopamine = 1.0f);

void launchRewardTraceUpdate(float* d_reward_traces,
                            int num_traces,
                            float decay_factor = 0.95f,
                            float current_reward = 0.0f);

#endif // ELIGIBILITY_AND_REWARD_KERNELS_CUH

// ============================================================================
// NeuronSpikingKernels.cuh - DECLARATIONS ONLY  
// ============================================================================

#ifndef NEURON_SPIKING_KERNELS_CUH
#define NEURON_SPIKING_KERNELS_CUH


// KERNEL DECLARATIONS (NO DEFINITIONS IN HEADER)

/**
 * Reset spike flags for all neurons
 */
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons);

/**
 * Process neuron spiking and update spike times
 */
__global__ void processNeuronSpikes(GPUNeuronState* neurons, 
                                   int* spike_counts,
                                   float current_time,
                                   int num_neurons);

/**
 * Handle dendritic spike propagation
 */
__global__ void dendriticSpikeKernel(GPUNeuronState* neurons, 
                                    float current_time, 
                                    int num_neurons);

/**
 * Update neuron voltages with leak currents
 */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons,
                                    float* I_leak,
                                    float* Cm,
                                    float dt,
                                    int num_neurons);

// WRAPPER FUNCTION DECLARATIONS

void launchResetSpikeFlags(GPUNeuronState* d_neurons, int num_neurons);

void launchProcessNeuronSpikes(GPUNeuronState* d_neurons,
                              int* d_spike_counts,
                              float current_time,
                              int num_neurons);

void launchDendriticSpikeKernel(GPUNeuronState* d_neurons, 
                               float current_time,
                               int num_neurons);

void launchUpdateNeuronVoltages(GPUNeuronState* d_neurons,
                               float* d_I_leak,
                               float* d_Cm,
                               float dt,
                               int num_neurons);

#endif // NEURON_SPIKING_KERNELS_CUH