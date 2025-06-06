#ifndef STDP_KERNEL_CUH
#define STDP_KERNEL_CUH

#include <cuda_runtime.h>

// Forward declarations
struct GPUSynapse;
struct GPUNeuronState;

/**
 * CUDA kernel for STDP weight updates with reward modulation
 * @param d_synapses Array of synapses on device
 * @param d_neurons Array of neuron states on device
 * @param num_synapses Number of synapses
 * @param A_plus Potentiation amplitude
 * @param A_minus Depression amplitude
 * @param tau_plus Potentiation time constant
 * @param tau_minus Depression time constant
 * @param current_time Current simulation time
 * @param min_weight Minimum allowed weight
 * @param max_weight Maximum allowed weight
 * @param reward_signal Reward modulation signal
 */
__global__ void stdpUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                                int num_synapses, float A_plus, float A_minus,
                                float tau_plus, float tau_minus, float eligibility_decay,
                                float learning_rate, float current_time,
                                float min_weight, float max_weight, float reward_signal);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Launch wrapper for STDP update kernel
 * @param d_synapses Array of synapses on device
 * @param d_neurons Array of neuron states on device
 * @param num_synapses Number of synapses
 * @param A_plus Potentiation amplitude
 * @param A_minus Depression amplitude
 * @param tau_plus Potentiation time constant
 * @param tau_minus Depression time constant
 * @param current_time Current simulation time
 * @param min_weight Minimum allowed weight
 * @param max_weight Maximum allowed weight
 * @param reward_signal Reward modulation signal
 */
void launchSTDPUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                           int num_synapses, float A_plus, float A_minus,
                           float tau_plus, float tau_minus, float eligibility_decay,
                           float learning_rate, float current_time,
                           float min_weight, float max_weight, float reward_signal);

#ifdef __cplusplus
}
#endif

#endif // STDP_KERNEL_CUH