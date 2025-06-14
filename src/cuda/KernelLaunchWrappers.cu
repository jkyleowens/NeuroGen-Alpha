// CUDA Compatibility and type trait fixes
#include <NeuroGen/cuda/CudaCompatibility.h>
#include <NeuroGen/cuda/CudaUtils.h>

#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/cuda/SynapseInputKernel.cuh>
#include <NeuroGen/cuda/EnhancedSTDPKernel.cuh>
#include <NeuroGen/cuda/RandomStateInit.cuh>
#include <NeuroGen/cuda/GridBlockUtils.cuh>
#include <NeuroGen/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// Include the cortical column definition
#ifdef __CUDACC__
#include "../cuda/CorticalColumn.h"
#else
#include "../GPUStructuresFwd.h"
#endif

// Forward declarations for missing kernels
__global__ void updateNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons, float dt, float current_time);
__global__ void initializeNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons);
__global__ void initializeSynapseStatesKernel(GPUSynapse* synapses, int num_synapses);
__global__ void resetNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons);
__global__ void updateSynapseStatesKernel(GPUSynapse* synapses, int num_synapses, float dt);
__global__ void processSpikesKernel(GPUNeuronState* neurons, int* spike_counts, float current_time, int num_neurons);
__global__ void applyInputCurrentsKernel(GPUNeuronState* neurons, const float* input_data, int input_size, int num_neurons);
__global__ void processSynapticInputsKernel(GPUNeuronState* neurons, GPUSynapse* synapses, int num_synapses, int num_neurons);
__global__ void updateEligibilityTracesKernel(GPUSynapse* synapses, GPUNeuronState* neurons, float dt, int num_synapses);
__global__ void applyHebbianLearningKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses);
__global__ void applyRewardModulationKernel(GPUSynapse* synapses, float reward, int num_synapses);
__global__ void applyHomeostaticScalingKernel(GPUSynapse* synapses, int num_synapses);
__global__ void initializeRandomStatesKernel(curandState* states, int num_states, unsigned long seed);
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons);
__global__ void extractOutputImproved(const GPUNeuronState* neurons, float* output_buffer,
                                     int output_size, float current_time);
__global__ void injectInputCurrentImproved(GPUNeuronState* neurons, const float* input_data, 
                                          int input_size, float current_time, float scale);
__global__ void applyRewardModulationImproved(GPUNeuronState* neurons, int num_neurons, float reward);
__global__ void computeNetworkStatistics(const GPUNeuronState* neurons, const GPUSynapse* synapses,
                                        int num_neurons, int num_synapses, float* stats);

// Test kernel to validate memory access
__global__ void testMemoryAccessKernel(GPUNeuronState* neurons, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons || idx < 0 || neurons == nullptr) return;
    
    // Simple memory access test
    neurons[idx].voltage = -65.0f;
    neurons[idx].spiked = false;
    neurons[idx].active = 1;
}

// =====================================================
// WRAPPER FUNCTION IMPLEMENTATIONS
// =====================================================

// Fallback wrapper for input current application
void applyInputCurrentsWrapper(GPUNeuronState* d_neurons, 
                               const float* input_data, 
                               int input_size,
                               int num_neurons) {
    // Validate inputs
    if (!d_neurons || !input_data || input_size <= 0 || num_neurons <= 0) {
        printf("[ERROR] Invalid parameters to applyInputCurrentsWrapper\n");
        return;
    }
    
    if (input_size > num_neurons) {
        printf("[ERROR] Input size (%d) exceeds neuron count (%d)\n", input_size, num_neurons);
        return;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA error before kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    
    dim3 block(256);
    dim3 grid((input_size + block.x - 1) / block.x);
    
    printf("[DEBUG] Launching input kernel: grid(%d,%d,%d) block(%d,%d,%d) input_size=%d num_neurons=%d\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, input_size, num_neurons);
    
    applyInputCurrentsKernel<<<grid, block>>>(d_neurons, input_data, input_size, num_neurons);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("[DEBUG] Input kernel completed successfully\n");
}

// Wrapper for memory test
void testMemoryAccessWrapper(GPUNeuronState* d_neurons, int num_neurons) {
    if (!d_neurons || num_neurons <= 0) {
        printf("[ERROR] Invalid parameters for memory test\n");
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    printf("[DEBUG] Testing memory access for %d neurons\n", num_neurons);
    
    testMemoryAccessKernel<<<grid, block>>>(d_neurons, num_neurons);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] Memory test failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("[DEBUG] Memory test passed\n");
    }
}

// Fallback wrapper for synaptic input processing
void processSynapticInputsWrapper(GPUNeuronState* d_neurons, 
                                  GPUSynapse* d_synapses, 
                                  int num_synapses,
                                  int num_neurons) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    processSynapticInputsKernel<<<grid, block>>>(d_neurons, d_synapses, num_synapses, num_neurons);
    cudaDeviceSynchronize();
}

// Fallback wrapper for eligibility trace updates
void updateEligibilityTracesWrapper(dim3 blocks, dim3 threads,
                                    GPUSynapse* d_synapses,
                                    GPUNeuronState* d_neurons,
                                    float dt_ms,
                                    int num_synapses) {
    updateEligibilityTracesKernel<<<blocks, threads>>>(d_synapses, d_neurons, dt_ms, num_synapses);
    cudaDeviceSynchronize();
}

// Fallback wrapper for Hebbian learning
void applyHebbianLearningWrapper(dim3 blocks, dim3 threads,
                                 GPUSynapse* d_synapses,
                                 GPUNeuronState* d_neurons,
                                 int num_synapses) {
    applyHebbianLearningKernel<<<blocks, threads>>>(d_synapses, d_neurons, num_synapses);
    cudaDeviceSynchronize();
}

// Fallback wrapper for reward modulation
void applyRewardModulationWrapper(dim3 blocks, dim3 threads,
                                  GPUSynapse* d_synapses,
                                  float reward,
                                  int num_synapses) {
    applyRewardModulationKernel<<<blocks, threads>>>(d_synapses, reward, num_synapses);
    cudaDeviceSynchronize();
}

// Fallback wrapper for homeostatic scaling
void applyHomeostaticScalingWrapper(dim3 blocks, dim3 threads,
                                    GPUSynapse* d_synapses,
                                    int num_synapses) {
    applyHomeostaticScalingKernel<<<blocks, threads>>>(d_synapses, num_synapses);
    cudaDeviceSynchronize();
}

// Fallback wrapper for neuron state updates
void updateNeuronStatesWrapper(dim3 blocks, dim3 threads,
                               GPUNeuronState* d_neurons,
                               int num_neurons,
                               float dt_ms,
                               float current_time_ms) {
    updateNeuronStatesKernel<<<blocks, threads>>>(d_neurons, num_neurons, dt_ms, current_time_ms);
    cudaDeviceSynchronize();
}

// Fallback wrapper for spike processing
void processSpikesWrapper(dim3 blocks, dim3 threads,
                          GPUNeuronState* d_neurons,
                          int* d_spike_counts,
                          float current_time_ms,
                          int num_neurons) {
    processSpikesKernel<<<blocks, threads>>>(d_neurons, d_spike_counts, current_time_ms, num_neurons);
    cudaDeviceSynchronize();
}

// Fallback wrapper for neuron state initialization
void initializeNeuronStatesWrapper(dim3 blocks, dim3 threads,
                                   GPUNeuronState* d_neurons,
                                   int num_neurons) {
    initializeNeuronStatesKernel<<<blocks, threads>>>(d_neurons, num_neurons);
    cudaDeviceSynchronize();
}

// Fallback wrapper for synapse state initialization
void initializeSynapseStatesWrapper(dim3 blocks, dim3 threads,
                                    GPUSynapse* d_synapses,
                                    int num_synapses) {
    initializeSynapseStatesKernel<<<blocks, threads>>>(d_synapses, num_synapses);
    cudaDeviceSynchronize();
}

// Fallback wrapper for random state initialization
void initializeRandomStatesWrapper(dim3 blocks, dim3 threads,
                                   curandState* d_states,
                                   int num_states,
                                   unsigned long seed) {
    initializeRandomStatesKernel<<<blocks, threads>>>(d_states, num_states, seed);
    cudaDeviceSynchronize();
}

// Fallback wrapper for cortical column initialization
void initializeCorticalColumnsWrapper(GPUCorticalColumn* d_columns,
                                      int num_columns) {
    // Simple memset for initialization
    cudaMemset(d_columns, 0, num_columns * sizeof(GPUCorticalColumn));
    cudaDeviceSynchronize();
}

// Fallback wrapper for neuron state reset
void resetNeuronStatesWrapper(dim3 blocks, dim3 threads,
                              GPUNeuronState* d_neurons,
                              int num_neurons) {
    resetNeuronStatesKernel<<<blocks, threads>>>(d_neurons, num_neurons);
    cudaDeviceSynchronize();
}

// Fallback wrapper for synapse state updates
void updateSynapseStatesWrapper(dim3 blocks, dim3 threads,
                                GPUSynapse* d_synapses,
                                int num_synapses,
                                float dt_ms) {
    updateSynapseStatesKernel<<<blocks, threads>>>(d_synapses, num_synapses, dt_ms);
    cudaDeviceSynchronize();
}

// =====================================================
// MAIN KERNEL LAUNCH FUNCTIONS
// =====================================================

extern "C" void launchUpdateNeuronVoltages(GPUNeuronState* neurons, float* I_leak, float* Cm, float dt, int N) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(N);
    updateNeuronVoltages<<<grid, block>>>(neurons, I_leak, Cm, dt, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchNeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, 0.0f, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchSynapseInputKernelInternal(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_synapses);
    synapseInputKernel<<<grid, block>>>(d_synapses, d_neurons, num_synapses);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

// C-linkage version
extern "C" void launchSynapseInputKernel(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses) {
    launchSynapseInputKernelInternal(d_synapses, d_neurons, num_synapses);
}

void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt, float current_time) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, current_time, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchSpikeDetectionKernel(GPUNeuronState* neurons, GPUSpikeEvent* spikes, float threshold,
                                int* spike_count, int num_neurons, float current_time) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(num_neurons, 256);
    detectSpikes<<<grid, block>>>(neurons, spikes, threshold, spike_count, num_neurons, current_time);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

void launchDendriticSpikeKernel(GPUNeuronState* neurons, int N, float current_time) {
    dim3 block = makeSafeBlock(256);
    dim3 grid = makeSafeGrid(N, 256);
    dendriticSpikeKernel<<<grid, block>>>(neurons, current_time, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately
    }
    cudaDeviceSynchronize();
}

// =====================================================
// STUB KERNEL IMPLEMENTATIONS
// =====================================================

__global__ void updateNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons, float dt, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Placeholder implementation with bounds checking
    if (idx < num_neurons) {
        neurons[idx].voltage += dt * 0.01f;
    }
}

__global__ void initializeNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Basic initialization
    neurons[idx].voltage = -70.0f;
    neurons[idx].spiked = false;
}

__global__ void initializeSynapseStatesKernel(GPUSynapse* synapses, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Basic synapse initialization
    synapses[idx].active = 1;
}

__global__ void resetNeuronStatesKernel(GPUNeuronState* neurons, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Reset neuron states
    neurons[idx].spiked = false;
    neurons[idx].voltage = -70.0f;
}

__global__ void updateSynapseStatesKernel(GPUSynapse* synapses, int num_synapses, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder synapse update
    // Just mark as active
    synapses[idx].active = 1;
}

__global__ void processSpikesKernel(GPUNeuronState* neurons, int* spike_counts, float current_time, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Simple spike detection with bounds checking
    if (idx < num_neurons && neurons[idx].voltage > 0.0f) {
        neurons[idx].spiked = true;
        if (spike_counts) {
            atomicAdd(spike_counts, 1);
        }
    }
}

__global__ void applyInputCurrentsKernel(GPUNeuronState* neurons, const float* input_data, int input_size, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Robust bounds checking
    if (idx >= input_size || idx >= num_neurons || neurons == nullptr || input_data == nullptr) {
        return;
    }
    
    // Extra safety check - ensure indices are valid
    if (idx < 0 || idx >= min(input_size, num_neurons)) {
        return;
    }
    
    // Apply input current to neurons with additional safety
    float input_current = input_data[idx];
    if (isfinite(input_current)) {  // Check for NaN/Inf
        neurons[idx].voltage += input_current * 0.1f;
    }
}

__global__ void processSynapticInputsKernel(GPUNeuronState* neurons, GPUSynapse* synapses, int num_synapses, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder synaptic processing with bounds checking
    GPUSynapse& synapse = synapses[idx];
    if (synapse.active && synapse.post_neuron_idx >= 0 && synapse.post_neuron_idx < num_neurons) {
        // Simple synaptic transmission
        neurons[synapse.post_neuron_idx].voltage += synapse.weight * 0.01f;
    }
}

__global__ void updateEligibilityTracesKernel(GPUSynapse* synapses, GPUNeuronState* neurons, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder eligibility trace update
    // Just decay any existing traces
    // Note: This assumes eligibility_trace field exists in GPUSynapse
}

__global__ void applyHebbianLearningKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder Hebbian learning
    // Just maintain current weights
}

__global__ void applyRewardModulationKernel(GPUSynapse* synapses, float reward, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder reward modulation
    // Apply small weight change based on reward
    synapses[idx].weight += reward * 0.001f;
}

__global__ void applyHomeostaticScalingKernel(GPUSynapse* synapses, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Placeholder homeostatic scaling
    // Keep weights bounded
    if (synapses[idx].weight > 2.0f) synapses[idx].weight = 2.0f;
    if (synapses[idx].weight < 0.0f) synapses[idx].weight = 0.0f;
}

__global__ void initializeRandomStatesKernel(curandState* states, int num_states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    // Initialize curand state
    curand_init(seed, idx, 0, &states[idx]);
}

// Extract output kernel
__global__ void extractOutputImproved(const GPUNeuronState* neurons, float* output_buffer,
                                     int output_size, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    // Extract neuron activity as output
    if (idx < output_size) {
        const GPUNeuronState& neuron = neurons[idx];
        output_buffer[idx] = neuron.spiked ? 1.0f : 0.0f;
    }
}

// Inject input current kernel
__global__ void injectInputCurrentImproved(GPUNeuronState* neurons, const float* input_data, 
                                          int input_size, float current_time, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;
    
    GPUNeuronState& neuron = neurons[idx];
    if (neuron.active && idx < input_size) {
        // Inject input current by adding to voltage
        neuron.voltage += input_data[idx] * scale;
    }
}

// Apply reward modulation kernel
__global__ void applyRewardModulationImproved(GPUNeuronState* neurons, int num_neurons, float reward) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    if (neuron.active) {
        // Apply reward modulation to neuron excitability
        neuron.voltage += reward * 0.01f;
    }
}

// Compute network statistics kernel
__global__ void computeNetworkStatistics(const GPUNeuronState* neurons, const GPUSynapse* synapses,
                                        int num_neurons, int num_synapses, float* stats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple statistics computation
    __shared__ float local_voltage_sum[256];
    __shared__ int local_spike_count[256];
    
    int tid = threadIdx.x;
    local_voltage_sum[tid] = 0.0f;
    local_spike_count[tid] = 0;
    
    if (idx < num_neurons) {
        local_voltage_sum[tid] = neurons[idx].voltage;
        local_spike_count[tid] = neurons[idx].spiked ? 1 : 0;
    }
    
    __syncthreads();
    
    // Block reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_voltage_sum[tid] += local_voltage_sum[tid + stride];
            local_spike_count[tid] += local_spike_count[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&stats[0], local_voltage_sum[0]);  // Total voltage
        atomicAdd(&stats[1], (float)local_spike_count[0]);  // Total spikes
    }
}

// Apply homeostatic scaling kernel
__global__ void applyHomeostaticScalingKernel(GPUSynapse* synapses, int num_synapses, 
                                             float scale_factor, float target_rate, float current_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (synapse.active) {
        // Apply homeostatic scaling
        float scaling = target_rate / (current_rate + 1e-6f);
        synapse.weight *= (1.0f + (scaling - 1.0f) * scale_factor);
        
        // Clamp weights
        if (synapse.weight > 5.0f) synapse.weight = 5.0f;
        if (synapse.weight < 0.0f) synapse.weight = 0.0f;
    }
}
