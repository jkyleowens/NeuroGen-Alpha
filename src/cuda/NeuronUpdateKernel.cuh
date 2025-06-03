#ifndef NEURON_UPDATE_KERNEL_CUH
#define NEURON_UPDATE_KERNEL_CUH

#include <cuda_runtime.h>

// Forward declarations
struct GPUNeuronState;

/**
 * CUDA kernel for RK4 integration of Hodgkin-Huxley model
 * @param neurons Array of neuron states
 * @param dt Time step
 * @param N Number of neurons
 */
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N);

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

#endif // NEURON_UPDATE_KERNEL_CUH