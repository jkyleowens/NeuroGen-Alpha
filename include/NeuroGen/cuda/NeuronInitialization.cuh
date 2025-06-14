#ifndef NEURON_INITIALIZATION_CUH
#define NEURON_INITIALIZATION_CUH

#include <cuda_runtime.h>

// Forward declarations
struct GPUNeuronState;

/**
 * CUDA kernel for initializing neuron compartments
 * @param neurons Array of neuron states
 * @param N Number of neurons
 */
__global__ void initializeNeuronCompartments(GPUNeuronState* neurons, int N);

/**
 * CUDA kernel for resetting neuron state
 * @param neurons Array of neuron states
 * @param N Number of neurons
 */
__global__ void resetNeuronState(GPUNeuronState* neurons, int N);

#endif // NEURON_INITIALIZATION_CUH
