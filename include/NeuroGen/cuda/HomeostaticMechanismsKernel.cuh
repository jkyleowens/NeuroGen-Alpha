#ifndef HOMEOSTATIC_MECHANISMS_KERNEL_CUH
#define HOMEOSTATIC_MECHANISMS_KERNEL_CUH

#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <math.h>

// ========== KERNEL DECLARATIONS ONLY ==========

/**
 * @brief Computes a scaling factor for each neuron based on its firing rate.
 * The factor is stored on the neuron for a subsequent kernel to apply.
 */
__global__ void computeSynapticScalingFactorKernel(GPUNeuronState* neurons, float dt, int num_neurons);

/**
 * @brief Applies the scaling factors computed by the previous kernel to the synaptic weights.
 */
__global__ void applySynapticScalingKernel(GPUSynapse* synapses, const GPUNeuronState* neurons, int num_synapses);

/**
 * @brief Enforces total weight constraints for incoming and outgoing synapses.
 */
__global__ void weightNormalizationKernel(GPUSynapse* synapses,
                                         int* neuron_synapse_counts,
                                         int num_synapses,
                                         int num_neurons);

/**
 * @brief Adjusts neuron intrinsic properties to maintain target activity levels.
 */
__global__ void activityRegulationKernel(GPUNeuronState* neurons,
                                        float current_time,
                                        float dt,
                                        int num_neurons);

/**
 * @brief Computes global network statistics for high-level monitoring.
 */
__global__ void networkHomeostaticMonitoringKernel(const GPUNeuronState* neurons,
                                                  const GPUSynapse* synapses,
                                                  float* network_stats,
                                                  int num_neurons,
                                                  int num_synapses);

/**
 * @brief Applies a strong, global dampening effect if network activity becomes pathological.
 */
__global__ void emergencyStabilizationKernel(GPUSynapse* synapses,
                                            GPUNeuronState* neurons,
                                            float network_activity_level,
                                            float emergency_threshold,
                                            int num_synapses,
                                            int num_neurons);


#endif // HOMEOSTATIC_MECHANISMS_KERNEL_CUH