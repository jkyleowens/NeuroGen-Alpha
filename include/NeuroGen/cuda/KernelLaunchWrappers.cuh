#ifndef KERNEL_LAUNCH_WRAPPERS_CUH
#define KERNEL_LAUNCH_WRAPPERS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Forward declarations
struct GPUNeuronState;
struct GPUSynapse;
struct GPUSpikeEvent;

/**
 * Launch kernel to update neuron voltages
 * @param neurons Array of neuron states
 * @param I_leak Array of leak currents (optional)
 * @param Cm Array of membrane capacitances (optional)
 * @param dt Time step
 * @param N Number of neurons
 */
extern "C" void launchUpdateNeuronVoltages(GPUNeuronState* neurons, 
                                          float* I_leak, float* Cm, 
                                          float dt, int N);

/**
 * Launch kernel for RK4 integration of Hodgkin-Huxley model
 * @param neurons Array of neuron states
 * @param dt Time step
 * @param N Number of neurons
 */
void launchNeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N);

/**
 * Launch kernel for RK4 integration of Hodgkin-Huxley model (alternative signature)
 * @param neurons Array of neuron states
 * @param N Number of neurons
 * @param dt Time step
 */
void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt);

/**
 * Launch kernel for synapse input processing
 * @param d_synapses Array of synapses
 * @param d_neurons Array of neuron states
 * @param num_synapses Number of synapses
 */
extern "C" void launchSynapseInputKernel(GPUSynapse* d_synapses, 
                                        GPUNeuronState* d_neurons, 
                                        int num_synapses);

/**
 * Launch kernel for spike detection
 * @param neurons Array of neuron states
 * @param spikes Array to store spike events
 * @param threshold Voltage threshold for spike
 * @param spike_count Pointer to counter for number of spikes
 * @param num_neurons Number of neurons
 * @param current_time Current simulation time
 */
void launchSpikeDetectionKernel(GPUNeuronState* neurons, 
                               GPUSpikeEvent* spikes, 
                               float threshold,
                               int* spike_count, 
                               int num_neurons, 
                               float current_time);

#endif // KERNEL_LAUNCH_WRAPPERS_CUH