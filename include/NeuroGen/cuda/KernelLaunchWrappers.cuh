#ifndef KERNEL_LAUNCH_WRAPPERS_CUH
#define KERNEL_LAUNCH_WRAPPERS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Forward declarations
struct GPUNeuronState;
struct GPUSynapse;
struct GPUSpikeEvent;
struct GPUCorticalColumn;

// =====================================================
// MAIN KERNEL LAUNCH FUNCTIONS
// =====================================================

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
 * Internal version of synapse input kernel launch (non-C linkage)
 */
void launchSynapseInputKernelInternal(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses);

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

// =====================================================
// WRAPPER FUNCTIONS FOR COMPLEX OPERATIONS
// =====================================================

/**
 * @brief Wrapper for applying input currents to neurons
 * @param d_neurons Array of neuron states on device
 * @param input_data Input current data
 * @param input_size Number of input elements
 * @param num_neurons Total number of neurons (for bounds checking)
 */
void applyInputCurrentsWrapper(GPUNeuronState* d_neurons, 
                               const float* input_data, 
                               int input_size,
                               int num_neurons);

/**
 * @brief Wrapper for processing synaptic inputs
 * @param d_neurons Array of neuron states on device
 * @param d_synapses Array of synapses on device
 * @param num_synapses Number of synapses
 * @param num_neurons Total number of neurons (for bounds checking)
 */
void processSynapticInputsWrapper(GPUNeuronState* d_neurons, 
                                  GPUSynapse* d_synapses, 
                                  int num_synapses,
                                  int num_neurons);

/**
 * @brief Test memory access wrapper
 * @param d_neurons Array of neuron states on device
 * @param num_neurons Total number of neurons
 */
void testMemoryAccessWrapper(GPUNeuronState* d_neurons, int num_neurons);

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
void updateEligibilityTracesWrapper(dim3 blocks, dim3 threads, 
                                    GPUSynapse* d_synapses, 
                                    GPUNeuronState* d_neurons, 
                                    float dt_ms, 
                                    int num_synapses);

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
void applyRewardModulationWrapper(dim3 blocks, dim3 threads, 
                                  GPUSynapse* d_synapses, 
                                  float reward, 
                                  int num_synapses);

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
void applyHebbianLearningWrapper(dim3 blocks, dim3 threads, 
                                 GPUSynapse* d_synapses, 
                                 GPUNeuronState* d_neurons, 
                                 int num_synapses);

/**
 * @brief Wrapper for homeostatic scaling operations
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_synapses Array of synapses on device
 * @param num_synapses Number of synapses
 */
void applyHomeostaticScalingWrapper(dim3 blocks, dim3 threads,
                                    GPUSynapse* d_synapses,
                                    int num_synapses);

/**
 * @brief Wrapper for neuron state updates
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_neurons Array of neuron states on device
 * @param num_neurons Number of neurons
 * @param dt_ms Time step in milliseconds
 * @param current_time_ms Current simulation time in milliseconds
 */
void updateNeuronStatesWrapper(dim3 blocks, dim3 threads,
                               GPUNeuronState* d_neurons,
                               int num_neurons,
                               float dt_ms,
                               float current_time_ms);

/**
 * @brief Wrapper for spike processing
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_neurons Array of neuron states on device
 * @param d_spike_counts Spike count array on device
 * @param current_time_ms Current simulation time in milliseconds
 * @param num_neurons Number of neurons
 */
void processSpikesWrapper(dim3 blocks, dim3 threads,
                          GPUNeuronState* d_neurons,
                          int* d_spike_counts,
                          float current_time_ms,
                          int num_neurons);

/**
 * @brief Wrapper for neuron state initialization
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_neurons Array of neuron states on device
 * @param num_neurons Number of neurons
 */
void initializeNeuronStatesWrapper(dim3 blocks, dim3 threads,
                                   GPUNeuronState* d_neurons,
                                   int num_neurons);

/**
 * @brief Wrapper for synapse state initialization
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_synapses Array of synapses on device
 * @param num_synapses Number of synapses
 */
void initializeSynapseStatesWrapper(dim3 blocks, dim3 threads,
                                    GPUSynapse* d_synapses,
                                    int num_synapses);

/**
 * @brief Wrapper for random state initialization
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_states Random states array on device
 * @param num_states Number of random states
 * @param seed Random seed
 */
void initializeRandomStatesWrapper(dim3 blocks, dim3 threads,
                                   curandState* d_states,
                                   int num_states,
                                   unsigned long seed);

/**
 * @brief Wrapper for cortical column initialization
 * @param d_columns Array of cortical columns on device
 * @param num_columns Number of columns
 */
void initializeCorticalColumnsWrapper(GPUCorticalColumn* d_columns,
                                      int num_columns);

/**
 * @brief Wrapper for neuron state reset
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_neurons Array of neuron states on device
 * @param num_neurons Number of neurons
 */
void resetNeuronStatesWrapper(dim3 blocks, dim3 threads,
                              GPUNeuronState* d_neurons,
                              int num_neurons);

/**
 * @brief Wrapper for synapse state updates
 * @param blocks Grid dimensions for kernel launch
 * @param threads Block dimensions for kernel launch
 * @param d_synapses Array of synapses on device
 * @param num_synapses Number of synapses
 * @param dt_ms Time step in milliseconds
 */
void updateSynapseStatesWrapper(dim3 blocks, dim3 threads,
                                GPUSynapse* d_synapses,
                                int num_synapses,
                                float dt_ms);

#endif // KERNEL_LAUNCH_WRAPPERS_CUH
