#ifndef NEURON_UPDATE_KERNEL_CUH
#define NEURON_UPDATE_KERNEL_CUH

#include <cuda_runtime.h>
#include "NeuronModelConstants.h"

// Forward declarations
struct GPUNeuronState;

/**
 * CUDA kernel for RK4 integration of Hodgkin-Huxley model
 * @param neurons Array of neuron states
 * @param dt Time step
 * @param current_time Current simulation time
 * @param N Number of neurons
 */
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons, float dt, float current_time, int N);

/**
 * CUDA kernel for simple voltage update
 * @param neurons Array of neuron states
 * @param I_leak Array of leak currents (optional)
 * @param Cm Array of membrane capacitances (optional)
 * @param dt Time step
 * @param N Number of neurons
 */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons, 
                                    float* I_leak, float* Cm,
                                    float dt, int N);

/**
 * CUDA kernel for processing dendritic spikes
 * @param neurons Array of neuron states
 * @param current_time Current simulation time
 * @param N Number of neurons
 */
__global__ void dendriticSpikeKernel(GPUNeuronState* neurons, float current_time, int N);

/**
 * CUDA kernel for updating neuron activity levels
 * @param neurons Array of neuron states
 * @param dt Time step
 * @param N Number of neurons
 */
__global__ void updateActivityLevels(GPUNeuronState* neurons, float dt, int N);

// Helper functions for Hodgkin-Huxley model
__device__ float alpha_m(float v);
__device__ float beta_m(float v);
__device__ float alpha_h(float v);
__device__ float beta_h(float v);
__device__ float alpha_n(float v);
__device__ float beta_n(float v);

// Helper functions for ion channel models
__device__ float computeMgBlock(float v);
__device__ float steadyStateActivation(float v, float v_half, float k);
__device__ float calciumDependentActivation(float ca_conc, float ca_half, float hill_coef);

#endif // NEURON_UPDATE_KERNEL_CUH
