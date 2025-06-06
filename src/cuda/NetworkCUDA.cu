// NetworkCUDA.cu - Complete implementation with all compilation errors fixed
#include <NeuroGen/cuda/CudaCompatibility.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/CudaUtils.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkPresets.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/Network.h> // For NetworkStats
#include <NeuroGen/cuda/STDPKernel.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/RandomStateInit.cuh>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>
#include <fstream>
#include <stdexcept>

// CUDA includes for kernel launch support
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__managed__ Network::NetworkStats g_stats;

// CUDA utility functions
dim3 getOptimalBlockSize() {
    return dim3(256, 1, 1); // Standard block size for most kernels
}

dim3 getOptimalGridSize(int num_elements) {
    int blocks_needed = (num_elements + 255) / 256; // Round up division
    return dim3(blocks_needed, 1, 1);
}

// Safe CUDA wrapper functions
template<typename T>
void safeCudaMalloc(T** ptr, size_t count) {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

template<typename T>
void safeCudaMemcpy(T* dst, const T* src, size_t count, cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), kind);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}

template<typename T>
void safeCudaMemset(T* ptr, int value, size_t count) {
    cudaError_t err = cudaMemset(ptr, value, count * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memset failed: " + std::string(cudaGetErrorString(err)));
    }
}

using NetworkStats = Network::NetworkStats;

// Global network state
static NetworkConfig g_config;
static GPUNeuronState* d_neurons = nullptr;
static GPUSynapse* d_synapses = nullptr;
static GPUCorticalColumn* d_columns = nullptr;
static std::vector<GPUCorticalColumn> h_columns;
static int num_columns = 0;
static float* d_input_buffer = nullptr;
static float* d_output_buffer = nullptr;
static float* d_reward_buffer = nullptr;
static GPUSpikeEvent* d_spike_events = nullptr;
static int* d_spike_count = nullptr;
static curandState* d_rng_states = nullptr;

// Network topology tracking
static int total_neurons = 0;
static int total_synapses = 0;
static int input_start, input_end;
static int hidden_start, hidden_end;
static int output_start, output_end;

static float current_time = 0.0f;
static bool network_initialized = false;

// CUDA kernel implementations
__global__ void injectInputCurrentImproved(GPUNeuronState* neurons, const float* input_data, 
                                          int input_size, float current_time, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;
    
    // Apply scaled input current to neuron voltage
    float current = input_data[idx] * scale;
    neurons[idx].voltages[0] += current;
    
    // Clamp voltage to reasonable range
    if (neurons[idx].voltages[0] > 50.0f) neurons[idx].voltages[0] = 50.0f;
    if (neurons[idx].voltages[0] < -100.0f) neurons[idx].voltages[0] = -100.0f;
}

__global__ void extractOutputImproved(const GPUNeuronState* neurons, float* output_buffer,
                                     int output_size, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    // Convert voltage to output signal (shifted to positive range)
    float voltage = neurons[idx].voltage;
    output_buffer[idx] = (voltage + 70.0f) / 120.0f; // Normalize to [0,1] approximately
    
    // Clamp output
    if (output_buffer[idx] < 0.0f) output_buffer[idx] = 0.0f;
    if (output_buffer[idx] > 1.0f) output_buffer[idx] = 1.0f;
}

__global__ void applyRewardModulationImproved(GPUNeuronState* neurons, int num_neurons, float reward) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Apply small reward-based voltage modulation
    float modulation = reward * 0.5f; // Reduced from 0.1f to prevent instability
    neurons[idx].voltages[0] += modulation;
    
    // Keep voltage in bounds
    if (neurons[idx].voltages[0] > 50.0f) neurons[idx].voltages[0] = 50.0f;
    if (neurons[idx].voltages[0] < -100.0f) neurons[idx].voltages[0] = -100.0f;
}

__global__ void computeNetworkStatistics(const GPUNeuronState* neurons, const GPUSynapse* synapses,
                                        int num_neurons, int num_synapses, float* stats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only use first thread to compute stats to avoid race conditions
    if (idx == 0) {
        float spike_count = 0.0f;
        float avg_voltage = 0.0f;
        
        for (int i = 0; i < num_neurons; ++i) {
            if (neurons[i].spiked) spike_count += 1.0f;
            avg_voltage += neurons[i].voltage;
        }
        
        stats[0] = spike_count;
        stats[1] = avg_voltage / num_neurons;
        
        // Compute average weight
        if (num_synapses > 0) {
            float avg_weight = 0.0f;
            for (int i = 0; i < num_synapses; ++i) {
                avg_weight += fabsf(synapses[i].weight);
            }
            stats[2] = avg_weight / num_synapses;
        } else {
            stats[2] = 0.0f;
        }
    }
}

__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    neurons[idx].spiked = false;
}

// Initialize the enhanced neural network
void initializeNetwork() {
    if (network_initialized) {
        std::cout << "[WARNING] Network already initialized. Skipping..." << std::endl;
        return;
    }
    
    std::cout << "[INIT] Starting network initialization..." << std::endl;
    
    // Use trading-optimized configuration by default
    g_config = NetworkPresets::trading_optimized();
    
    g_config.print();
    
    // Calculate network dimensions
    total_neurons = g_config.input_size + g_config.hidden_size + g_config.output_size;
    
    // Set layer boundaries
    input_start = 0;
    input_end = g_config.input_size;
    hidden_start = g_config.input_size;
    hidden_end = g_config.input_size + g_config.hidden_size;
    output_start = g_config.input_size + g_config.hidden_size;
    output_end = total_neurons;

    // Build cortical column descriptors for hidden layer
    h_columns.clear();
    h_columns.reserve(g_config.numColumns);
    for (int c = 0; c < g_config.numColumns; ++c) {
        int ns = hidden_start + c * g_config.neuronsPerColumn;
        int ne = ns + g_config.neuronsPerColumn;
        GPUCorticalColumn col{};
        initGPUCorticalColumn(&col, ns, ne, c);
        h_columns.push_back(col);
    }
    num_columns = static_cast<int>(h_columns.size());
    
    std::cout << "[CUDA] Initializing network with " << total_neurons << " neurons..." << std::endl;
    std::cout << "[CUDA] Input: [" << input_start << ", " << input_end << ")" << std::endl;
    std::cout << "[CUDA] Hidden: [" << hidden_start << ", " << hidden_end << ")" << std::endl;
    std::cout << "[CUDA] Output: [" << output_start << ", " << output_end << ")" << std::endl;
    
    // Allocate GPU memory with error checking
    safeCudaMalloc(&d_neurons, total_neurons);
    safeCudaMalloc(&d_input_buffer, g_config.input_size);
    safeCudaMalloc(&d_output_buffer, g_config.output_size);
    safeCudaMalloc(&d_reward_buffer, 1);
    safeCudaMalloc(&d_spike_events, total_neurons * 10); // Buffer for multiple spikes
    safeCudaMalloc(&d_spike_count, 1);
    safeCudaMalloc(&d_rng_states, total_neurons);
    
    // Initialize neurons with proper HH resting state
    std::vector<GPUNeuronState> host_neurons(total_neurons);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> voltage_dist(-65.0f, 2.0f);
    std::uniform_real_distribution<float> gating_dist(0.0f, 0.02f);
    
    for (int i = 0; i < total_neurons; ++i) {
        auto& neuron = host_neurons[i];
        neuron.voltage = voltage_dist(gen);
        neuron.spiked = false;
        neuron.last_spike_time = -1000.0f;
        
        // Initialize HH gating variables near resting state
        neuron.m = 0.05f + gating_dist(gen);
        neuron.h = 0.60f + gating_dist(gen);
        neuron.n = 0.32f + gating_dist(gen);
        
        // Single compartment for now
        neuron.compartment_count = 1;
        neuron.voltages[0] = neuron.voltage;
        neuron.I_leak[0] = 0.0f;
        neuron.Cm[0] = 1.0f;
    }
    
    safeCudaMemcpy(d_neurons, host_neurons.data(), total_neurons, cudaMemcpyHostToDevice);
    
    // Generate improved network topology
    std::vector<GPUSynapse> host_synapses;
    NetworkCUDAInternal::createNetworkTopology(host_synapses, h_columns, gen);

    // Patch column synapse ranges
    size_t syn_cursor = 0;
    for (auto& col : h_columns) {
        col.synapse_start = static_cast<int>(syn_cursor);
        col.synapse_end   = static_cast<int>(syn_cursor + g_config.localFanOut * g_config.neuronsPerColumn);
        syn_cursor = col.synapse_end;
    }
    
    total_synapses = host_synapses.size();
    std::cout << "[CUDA] Created " << total_synapses << " synapses" << std::endl;
    
    // Copy synapses to GPU
    if (total_synapses > 0) {
        safeCudaMalloc(&d_synapses, total_synapses);
        safeCudaMemcpy(d_synapses, host_synapses.data(), total_synapses, cudaMemcpyHostToDevice);
    }

    // Copy cortical columns to GPU
    if (num_columns > 0) {
        safeCudaMalloc(&d_columns, num_columns);
        safeCudaMemcpy(d_columns, h_columns.data(), num_columns, cudaMemcpyHostToDevice);
    }
    
    // Initialize random states
    launchRandomStateInit(d_rng_states, total_neurons, rd());
    CUDA_CHECK_KERNEL();
    
    // Zero out buffers
    safeCudaMemset(d_input_buffer, 0, g_config.input_size);
    safeCudaMemset(d_output_buffer, 0, g_config.output_size);
    safeCudaMemset(d_spike_count, 0, 1);
    
    network_initialized = true;
    g_stats.reset();
    current_time = 0.0f;
    
    std::cout << "[CUDA] Network initialization complete!" << std::endl;
}

// Enhanced network topology creation
namespace NetworkCUDAInternal {
void createNetworkTopology(std::vector<GPUSynapse>& synapses,
                           const std::vector<GPUCorticalColumn>& columns,
                           std::mt19937& gen) {
    std::uniform_real_distribution<float> weight_dist(0.0f, g_config.weight_init_std);
    std::uniform_real_distribution<float> delay_dist(g_config.delay_min, g_config.delay_max);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    synapses.clear();
    synapses.reserve(total_neurons * 20); // Conservative estimate
    
    std::cout << "[TOPOLOGY] Creating network connections..." << std::endl;
    
    int connections_created = 0;
    
    // Input to Hidden connections
    for (int pre = input_start; pre < input_end; ++pre) {
        for (int post = hidden_start; post < hidden_end; ++post) {
            if (prob_dist(gen) < g_config.input_hidden_prob) {
                GPUSynapse syn{};
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(gen) * (prob_dist(gen) < g_config.exc_ratio ? 1.0f : -1.0f);
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                syn.last_active = 0.0f;
                syn.eligibility_trace = 0.0f;
                synapses.push_back(syn);
                connections_created++;
            }
        }
    }

    std::cout << "[TOPOLOGY] Created " << connections_created << " input->hidden connections" << std::endl;
    connections_created = 0;

    // Local recurrent connections within each cortical column
    for (const auto& col : columns) {
        for (int n = col.neuron_start; n < col.neuron_end; ++n) {
            bool is_exc = (n - col.neuron_start) < static_cast<int>(g_config.exc_ratio * g_config.neuronsPerColumn);
            for (int k = 0; k < g_config.localFanOut; ++k) {
                int tgt = col.neuron_start + static_cast<int>(prob_dist(gen) * g_config.neuronsPerColumn);
                if (tgt == n) {
                    tgt = ((tgt - col.neuron_start + 1) % g_config.neuronsPerColumn) + col.neuron_start;
                }

                GPUSynapse syn{};
                syn.pre_neuron_idx = n;
                syn.post_neuron_idx = tgt;
                syn.weight = is_exc ? weight_dist(gen) : -weight_dist(gen);
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                syn.last_active = 0.0f;
                syn.eligibility_trace = 0.0f;
                synapses.push_back(syn);
                connections_created++;
            }
        }
    }

    std::cout << "[TOPOLOGY] Created " << connections_created << " local column connections" << std::endl;
    connections_created = 0;
    
    // Hidden to Output connections (fully connected)
    for (int pre = hidden_start; pre < hidden_end; ++pre) {
        for (int post = output_start; post < output_end; ++post) {
            GPUSynapse syn;
            syn.pre_neuron_idx = pre;
            syn.post_neuron_idx = post;
            syn.weight = weight_dist(gen) * 0.3f; // Smaller initial weights for output
            syn.delay = delay_dist(gen);
            syn.last_pre_spike_time = -1000.0f;
            syn.activity_metric = 0.0f;
            syn.last_active = 0.0f;
            syn.eligibility_trace = 0.0f;
            synapses.push_back(syn);
            connections_created++;
        }
    }
    
    std::cout << "[TOPOLOGY] Created " << connections_created << " hidden->output connections" << std::endl;
    std::cout << "[TOPOLOGY] Total synapses: " << synapses.size() << std::endl;
}

} // namespace NetworkCUDAInternal

// Enhanced forward pass with better dynamics
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    if (!network_initialized) {
        throw std::runtime_error("Network not initialized. Call initializeNetwork() first.");
    }
    
    NetworkCUDAInternal::validateInputs(input, reward_signal);
    
    // Copy input to GPU
    safeCudaMemcpy(d_input_buffer, input.data(), input.size(), cudaMemcpyHostToDevice);
    
    // Store reward signal
    safeCudaMemcpy(d_reward_buffer, &reward_signal, 1, cudaMemcpyHostToDevice);
    
    // Calculate simulation steps
    float simulation_time = 10.0f; // Default simulation time in ms
    int simulation_steps = static_cast<int>(simulation_time / g_config.dt);
    simulation_steps = std::min(simulation_steps, 1000); // Safety limit
    
    // Reset spike flags
    dim3 block = getOptimalBlockSize();
    dim3 grid = getOptimalGridSize(total_neurons);
    resetSpikeFlags<<<grid, block>>>(d_neurons, total_neurons);
    CUDA_CHECK_KERNEL();
    
    // Simulation loop
    for (int step = 0; step < simulation_steps; ++step) {
        current_time += g_config.dt;
        
        // Inject input current (every few steps to maintain stimulation)
        if (step % 5 == 0) {
            dim3 input_grid = getOptimalGridSize(g_config.input_size);
            injectInputCurrentImproved<<<input_grid, block>>>(
                d_neurons + input_start, d_input_buffer, g_config.input_size, 
                current_time, g_config.input_current_scale
            );
            CUDA_CHECK_KERNEL();
        }
        
        // Update neuron dynamics with RK4
        launchRK4NeuronUpdateKernel(d_neurons, total_neurons, g_config.dt, current_time);
        CUDA_CHECK_KERNEL();
        
        // Process dendritic spikes
        launchDendriticSpikeKernel(d_neurons, total_neurons, current_time);
        CUDA_CHECK_KERNEL();
        
        // Detect spikes and update spike flags
        safeCudaMemset(d_spike_count, 0, 1);
        launchSpikeDetectionKernel(d_neurons, d_spike_events, g_config.spike_threshold,
                                   d_spike_count, total_neurons, current_time);
        CUDA_CHECK_KERNEL();
        
        // Propagate spikes through synapses
        if (total_synapses > 0) {
            launchSynapseInputKernel(d_synapses, d_neurons, total_synapses);
            CUDA_CHECK_KERNEL();
        }
        
        // Apply reward modulation periodically
        if (step % 10 == 0) {
            applyRewardModulationImproved<<<grid, block>>>(
                d_neurons, total_neurons, reward_signal
            );
            CUDA_CHECK_KERNEL();
        }
    }
    
    // Extract output with improved encoding
    std::vector<float> raw_output(g_config.output_size);
    dim3 output_grid = getOptimalGridSize(g_config.output_size);
    extractOutputImproved<<<output_grid, block>>>(
        d_neurons + output_start, d_output_buffer, g_config.output_size, current_time
    );
    CUDA_CHECK_KERNEL();
    
    safeCudaMemcpy(raw_output.data(), d_output_buffer, g_config.output_size, cudaMemcpyDeviceToHost);
    
    // Apply softmax for decision probabilities
    return NetworkCUDAInternal::applySoftmax(raw_output);
}

namespace NetworkCUDAInternal {

// Improved softmax with numerical stability
std::vector<float> applySoftmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    if (input.empty()) {
        return output;
    }
    
    float max_val = *std::max_element(input.begin(), input.end());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }
    
    if (sum_exp > 1e-10f) {
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] /= sum_exp;
        }
    } else {
        // Fallback to uniform distribution
        float uniform_prob = 1.0f / input.size();
        std::fill(output.begin(), output.end(), uniform_prob);
    }
    
    return output;
}

void validateInputs(const std::vector<float>& input, float reward_signal) {
    if (input.size() != static_cast<size_t>(g_config.input_size)) {
        throw std::invalid_argument("Input size mismatch: expected " + 
                                  std::to_string(g_config.input_size) + 
                                  ", got " + std::to_string(input.size()));
    }
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument("Non-finite input detected at index " + std::to_string(i));
        }
    }
    
    if (!std::isfinite(reward_signal)) {
        throw std::invalid_argument("Non-finite reward signal");
    }
}

void updateNetworkStatistics() {
    if (!network_initialized || total_neurons == 0) return;
    
    // Compute network statistics
    static float stats_buffer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    dim3 block = getOptimalBlockSize();
    dim3 grid = getOptimalGridSize(total_neurons);
    computeNetworkStatistics<<<grid, block>>>(
        d_neurons, d_synapses, total_neurons, total_synapses, stats_buffer
    );
    CUDA_CHECK_KERNEL();
    
    // Update global statistics
    g_stats.total_spikes = stats_buffer[0];
    g_stats.avg_weight = stats_buffer[2];
    
    if (g_stats.update_count % g_config.monitoring_interval == 0) {
        std::cout << "[STATS] Spikes: " << g_stats.total_spikes 
                  << ", Avg Weight: " << g_stats.avg_weight
                  << ", Reward: " << g_stats.reward_signal << std::endl;
    }
}

// Homeostatic scaling to prevent runaway dynamics
void applyHomeostaticScaling() {
    // Placeholder for homeostatic scaling implementation
    // This would help maintain network stability over long training periods
}
} // namespace NetworkCUDAInternal

// Network weight updates with reward modulation
void updateSynapticWeightsCUDA(float reward_signal) {
    if (!network_initialized) {
        std::cout << "[WARNING] Network not initialized. Skipping weight update." << std::endl;
        return;
    }
    
    if (total_synapses == 0) {
        return;
    }
    
    // Adaptive STDP parameters based on reward
    float reward_factor = 1.0f + g_config.reward_learning_rate * reward_signal;
    float A_plus = g_config.A_plus * reward_factor;
    float A_minus = g_config.A_minus * (2.0f - reward_factor); // Inverse for depression
    
    // Apply STDP with eligibility traces and reward modulation
    launchSTDPUpdateKernel(d_synapses, d_neurons, total_synapses,
                           A_plus, A_minus, g_config.tau_plus, g_config.tau_minus,
                           g_config.eligibility_decay, g_config.reward_learning_rate,
                           current_time, g_config.min_weight, g_config.max_weight,
                           reward_signal);
    CUDA_CHECK_KERNEL();
    
    // Homeostatic mechanisms every 100 updates
    static int update_counter = 0;
    if (++update_counter % 100 == 0 && g_config.homeostatic_strength > 0) {
        NetworkCUDAInternal::applyHomeostaticScaling();
    }
}

void setNetworkConfig(const NetworkConfig& config) {
    if (network_initialized) {
        std::cout << "[WARNING] Cannot change config after initialization. Reset network first." << std::endl;
        return;
    }
    
    g_config = config;
    if (!g_config.validate()) {
        std::cerr << "[ERROR] Invalid network configuration provided" << std::endl;
        throw std::runtime_error("Invalid network configuration");
    }
}

NetworkConfig getNetworkConfig() {
    return g_config;
}

void printNetworkStats() {
    std::cout << "\n=== Network Statistics ===" << std::endl;
    std::cout << "Initialized: " << (network_initialized ? "YES" : "NO") << std::endl;
    std::cout << "Total Neurons: " << total_neurons << std::endl;
    std::cout << "Total Synapses: " << total_synapses << std::endl;
    std::cout << "Input Layer: [" << input_start << ", " << input_end << ")" << std::endl;
    std::cout << "Hidden Layer: [" << hidden_start << ", " << hidden_end << ")" << std::endl;
    std::cout << "Output Layer: [" << output_start << ", " << output_end << ")" << std::endl;
    std::cout << "Current Time: " << current_time << " ms" << std::endl;
    std::cout << "Update Count: " << g_stats.update_count << std::endl;
    std::cout << "Total Spikes: " << g_stats.total_spikes << std::endl;
    std::cout << "Average Weight: " << g_stats.avg_weight << std::endl;
    std::cout << "Last Reward Signal: " << g_stats.reward_signal << std::endl;
    std::cout << "=========================" << std::endl;
}

NetworkStats getNetworkStats() {
    return g_stats;
}

// Advanced features
void saveNetworkState(const std::string& filename) {
    if (!network_initialized) {
        std::cerr << "[ERROR] Network not initialized. Cannot save state." << std::endl;
        return;
    }
    
    std::cout << "[SAVE] Saving network state to " << filename << std::endl;
    
    // Copy GPU data to host
    std::vector<GPUNeuronState> host_neurons(total_neurons);
    std::vector<GPUSynapse> host_synapses(total_synapses);
    
    if (total_neurons > 0) {
        safeCudaMemcpy(host_neurons.data(), d_neurons, total_neurons, cudaMemcpyDeviceToHost);
    }
    if (total_synapses > 0) {
        safeCudaMemcpy(host_synapses.data(), d_synapses, total_synapses, cudaMemcpyDeviceToHost);
    }
    
    // Simple binary save
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    // Save metadata
    file.write(reinterpret_cast<const char*>(&total_neurons), sizeof(int));
    file.write(reinterpret_cast<const char*>(&total_synapses), sizeof(int));
    file.write(reinterpret_cast<const char*>(&current_time), sizeof(float));
    
    // Save neurons and synapses
    if (total_neurons > 0) {
        file.write(reinterpret_cast<const char*>(host_neurons.data()), 
                   total_neurons * sizeof(GPUNeuronState));
    }
    if (total_synapses > 0) {
        file.write(reinterpret_cast<const char*>(host_synapses.data()), 
                   total_synapses * sizeof(GPUSynapse));
    }
    
    file.close();
    std::cout << "[SAVE] Network state saved successfully" << std::endl;
}

void loadNetworkState(const std::string& filename) {
    std::cout << "[LOAD] Loading network state from " << filename << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file for reading: " << filename << std::endl;
        return;
    }
    
    // Load metadata
    int loaded_neurons, loaded_synapses;
    file.read(reinterpret_cast<char*>(&loaded_neurons), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_synapses), sizeof(int));
    file.read(reinterpret_cast<char*>(&current_time), sizeof(float));
    
    if (loaded_neurons != total_neurons || loaded_synapses != total_synapses) {
        std::cerr << "[WARNING] Network size mismatch." << std::endl;
        file.close();
        return;
    }
    
    // Load neurons and synapses
    std::vector<GPUNeuronState> host_neurons(total_neurons);
    std::vector<GPUSynapse> host_synapses(total_synapses);
    
    if (total_neurons > 0) {
        file.read(reinterpret_cast<char*>(host_neurons.data()), 
                  total_neurons * sizeof(GPUNeuronState));
        safeCudaMemcpy(d_neurons, host_neurons.data(), total_neurons, cudaMemcpyHostToDevice);
    }
    
    if (total_synapses > 0) {
        file.read(reinterpret_cast<char*>(host_synapses.data()), 
                  total_synapses * sizeof(GPUSynapse));
        safeCudaMemcpy(d_synapses, host_synapses.data(), total_synapses, cudaMemcpyHostToDevice);
    }
    
    file.close();
    std::cout << "[LOAD] Network state loaded successfully" << std::endl;
}

void resetNetwork() {
    std::cout << "[RESET] Resetting network to initial state" << std::endl;
    
    if (!network_initialized) {
        std::cout << "[WARNING] Network not initialized. Nothing to reset." << std::endl;
        return;
    }
    
    // Reset global state
    current_time = 0.0f;
    g_stats.reset();
    
    // Reinitialize neurons to resting state
    if (total_neurons > 0) {
        std::vector<GPUNeuronState> host_neurons(total_neurons);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> voltage_dist(-65.0f, 2.0f);
        std::uniform_real_distribution<float> gating_dist(0.0f, 0.02f);
        
        for (int i = 0; i < total_neurons; ++i) {
            auto& neuron = host_neurons[i];
            neuron.voltage = voltage_dist(gen);
            neuron.spiked = false;
            neuron.last_spike_time = -1000.0f;
            
            // Reset HH gating variables to resting state
            neuron.m = 0.05f + gating_dist(gen);
            neuron.h = 0.60f + gating_dist(gen);
            neuron.n = 0.32f + gating_dist(gen);
            
            // Reset compartments
            neuron.compartment_count = 1;
            neuron.voltages[0] = neuron.voltage;
            neuron.I_leak[0] = 0.0f;
            neuron.Cm[0] = 1.0f;
        }
        
        safeCudaMemcpy(d_neurons, host_neurons.data(), total_neurons, cudaMemcpyHostToDevice);
    }
    
    // Clear buffers
    if (d_input_buffer) safeCudaMemset(d_input_buffer, 0, g_config.input_size);
    if (d_output_buffer) safeCudaMemset(d_output_buffer, 0, g_config.output_size);
    if (d_spike_count) safeCudaMemset(d_spike_count, 0, 1);
    
    std::cout << "[RESET] Network reset complete" << std::endl;
}

void cleanupNetwork() {
    std::cout << "[CLEANUP] Cleaning up CUDA network..." << std::endl;
    
    if (d_neurons) { cudaFree(d_neurons); d_neurons = nullptr; }
    if (d_synapses) { cudaFree(d_synapses); d_synapses = nullptr; }
    if (d_input_buffer) { cudaFree(d_input_buffer); d_input_buffer = nullptr; }
    if (d_output_buffer) { cudaFree(d_output_buffer); d_output_buffer = nullptr; }
    if (d_reward_buffer) { cudaFree(d_reward_buffer); d_reward_buffer = nullptr; }
    if (d_spike_events) { cudaFree(d_spike_events); d_spike_events = nullptr; }
    if (d_spike_count) { cudaFree(d_spike_count); d_spike_count = nullptr; }
    if (d_rng_states) { cudaFree(d_rng_states); d_rng_states = nullptr; }
    
    total_neurons = 0;
    total_synapses = 0;
    network_initialized = false;
    
    std::cout << "[CLEANUP] Network cleanup complete" << std::endl;
}
