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
 * @param current_time Current simulation time
 */
void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt, float current_time);

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

/**
 * Launch kernel for processing dendritic spikes
 * @param neurons Array of neuron states
 * @param N Number of neurons
 * @param current_time Current simulation time
 */
void launchDendriticSpikeKernel(GPUNeuronState* neurons, int N, float current_time);

/**
 * @brief Launches the CUDA kernel to update the eligibility traces for all synapses.
 *
 * Eligibility traces are a fundamental component of reinforcement learning in SNNs,
 * marking synapses that have recently contributed to a neuron's firing.
 *
 * @param blocks The grid dimensions for the kernel launch.
 * @param threads The block dimensions for the kernel launch.
 * @param d_synapses Pointer to the synapse data on the GPU.
 * @param d_neurons Pointer to the neuron state data on the GPU.
 * @param dt_ms The simulation time step in milliseconds.
 * @param num_synapses The total number of synapses.
 */
void updateEligibilityTracesWrapper(dim3 blocks, dim3 threads, GPUSynapse* d_synapses, const GPUNeuronState* d_neurons, float dt_ms, int num_synapses);

/**
 * @brief Launches the CUDA kernel to apply a global reward signal to the synapses.
 *
 * This function modulates the weights of eligible synapses based on the reward signal,
 * reinforcing or weakening connections to steer the network toward desired behaviors.
 *
 * @param blocks The grid dimensions for the kernel launch.
 * @param threads The block dimensions for the kernel launch.
 * @param d_synapses Pointer to the synapse data on the GPU.
 * @param reward The global reward signal.
 * @param num_synapses The total number of synapses.
 */
void applyRewardModulationWrapper(dim3 blocks, dim3 threads, GPUSynapse* d_synapses, float reward, int num_synapses);

/**
 * @brief Launches the CUDA kernel to apply Hebbian learning rules (STDP).
 *
 * This implements Spike-Timing-Dependent Plasticity, a biologically plausible
 * learning rule where the precise timing of pre- and post-synaptic spikes
 * determines the change in synaptic strength.
 *
 * @param blocks The grid dimensions for the kernel launch.
 * @param threads The block dimensions for the kernel launch.
 * @param d_synapses Pointer to the synapse data on the GPU.
 * @param d_neurons Pointer to the neuron state data on the GPU.
 * @param num_synapses The total number of synapses.
 */
void applyHebbianLearningWrapper(dim3 blocks, dim3 threads, GPUSynapse* d_synapses, const GPUNeuronState* d_neurons, int num_synapses);


#endif // KERNEL_LAUNCH_WRAPPERS_CUH
