#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>
#include <cuda_runtime.h>

/**
 * @brief Resets the `spiked` flag for all neurons at the start of a simulation step.
 */
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    neurons[idx].spiked = false;
}

__global__ void countSpikesKernel(const GPUNeuronState* neurons,
                                 int* spike_count, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    if (neurons[idx].spiked) {
        atomicAdd(spike_count, 1);
    }
}

__global__ void updateNeuronSpikes(GPUNeuronState* neurons,
                                  float threshold, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Check for spike
    if (neuron.voltage >= threshold && !neuron.spiked) {
        neuron.spiked = true;
    } else {
        neuron.spiked = false;
    }
}

__global__ void detectSpikes(const GPUNeuronState* neurons,
                            GPUSpikeEvent* spikes, float threshold,
                            int* spike_count, int num_neurons, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = neurons[idx];
    
    // Check for spike
    if (neuron.voltage >= threshold) {
        // Record spike event
        int spike_idx = atomicAdd(spike_count, 1);
        
        // Ensure we don't overflow the spike buffer
        if (spike_idx < num_neurons * 10) {
            spikes[spike_idx].neuron_idx = idx;
            spikes[spike_idx].time = current_time;
            spikes[spike_idx].amplitude = 1.0f;
        }
    }
}

/**
 * @brief Update neuron voltages with leak currents
 */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons,
                                    float* I_leak,
                                    float* Cm,
                                    float dt,
                                    int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip if in refractory period
    if (neuron.refractory_period > 0.0f) {
        return;
    }
    
    // Update voltage for each compartment
    for (int c = 0; c < neuron.compartment_count; c++) {
        float leak_current = I_leak ? I_leak[idx] : -0.1f; // Default leak
        float membrane_capacitance = Cm ? Cm[idx] : 1.0f;  // Default capacitance
        
        // Simple voltage update: dV/dt = I_leak / Cm
        float dV = (leak_current / membrane_capacitance) * dt;
        neuron.voltages[c] += dV;
        
        // Clamp voltage to reasonable bounds
        neuron.voltages[c] = fmaxf(-100.0f, fminf(50.0f, neuron.voltages[c]));
    }
}

/**
 * @brief Handle dendritic spike propagation
 */
__global__ void dendriticSpikeKernel(GPUNeuronState* neurons, 
                                    float current_time, 
                                    int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Check each dendritic compartment for spike threshold
    for (int c = 1; c < neuron.compartment_count; c++) { // Skip soma (c=0)
        if (neuron.voltages[c] > neuron.dendritic_threshold[c]) {
            neuron.dendritic_spike[c] = true;
            neuron.dendritic_spike_time[c] = current_time;
            
            // Reset dendritic voltage
            neuron.voltages[c] = V_RESET;
            
            // Propagate spike to soma (simplified)
            float propagation_strength = 0.1f; // Adjustable parameter
            neuron.voltages[0] += propagation_strength * 10.0f; // Add depolarization to soma
        }
    }
}

// ============================================================================
// WRAPPER FUNCTION IMPLEMENTATIONS
// ============================================================================

void launchUpdateNeuronVoltages(GPUNeuronState* d_neurons,
                               float* d_I_leak,
                               float* d_Cm,
                               float dt,
                               int num_neurons) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    updateNeuronVoltages<<<grid, block>>>(d_neurons, d_I_leak, d_Cm, dt, num_neurons);
    cudaDeviceSynchronize();
}

void launchDendriticSpikeKernel(GPUNeuronState* d_neurons, 
                               float current_time,
                               int num_neurons) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    dendriticSpikeKernel<<<grid, block>>>(d_neurons, current_time, num_neurons);
    cudaDeviceSynchronize();
}