#ifndef ENHANCED_STDP_KERNEL_CUH
#define ENHANCED_STDP_KERNEL_CUH

#include <NeuroGen/cuda/GPUNeuralStructures.h>

/**
 * @brief Main kernel for multi-factor, biologically-inspired synaptic plasticity.
 *
 * This kernel calculates the potential for synaptic change (LTP or LTD) based on
 * several factors: precise spike timing, local calcium concentration, and the
 * current state of the synapse. It updates the fast eligibility trace, which is
 * later consolidated by the reward modulation kernel.
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
);

#endif // ENHANCED_STDP_KERNEL_CUH