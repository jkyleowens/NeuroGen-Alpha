#include "../../include/NeuroGen/cuda/NetworkCUDA.cuh"
#include "../../include/NeuroGen/cuda/CudaUtils.cuh"
#include "../../include/NeuroGen/NetworkConfig.h"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/STDPKernel.cuh"
#include "../../include/NeuroGen/cuda/KernelLaunchWrappers.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>
#include <curand_kernel.h>

// Global network state
static NetworkConfig g_config;
static GPUNeuronState* d_neurons = nullptr;
static GPUSynapse* d_synapses = nullptr;
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

// Performance monitoring
struct NetworkStats {
    float avg_firing_rate = 0.0f;
    float total_spikes = 0.0f;
    float avg_weight = 0.0f;
    float reward_signal = 0.0f;
    int update_count = 0;
    
    void reset() {
        avg_firing_rate = 0.0f;
        total_spikes = 0.0f;
        avg_weight = 0.0f;
        reward_signal = 0.0f;
    }
};

static NetworkStats g_stats;
static float current_time = 0.0f;

// Forward declarations for CUDA kernels
__global__ void injectInputCurrentImproved(GPUNeuronState* neurons, const float* input_data, 
                                          int input_size, float current_time, float scale);
__global__ void extractOutputImproved(const GPUNeuronState* neurons, float* output_buffer,
                                     int output_size, float current_time);
__global__ void applyRewardModulationImproved(GPUNeuronState* neurons, int num_neurons, float reward);
__global__ void computeNetworkStatistics(const GPUNeuronState* neurons, const GPUSynapse* synapses,
                                        int num_neurons, int num_synapses, float* stats);
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons);

// Initialize the enhanced neural network
void initializeNetwork() {
    // Use trading-optimized configuration by default
    g_config = NetworkPresets::trading_optimized();
    
    if (!g_config.validate()) {
        throw std::runtime_error("Invalid network configuration");
    }
    
    g_config.print();
    printDeviceInfo();
    
    // Calculate network dimensions
    total_neurons = g_config.input_size + g_config.hidden_size + g_config.output_size;
    
    // Set layer boundaries
    input_start = 0;
    input_end = g_config.input_size;
    hidden_start = g_config.input_size;
    hidden_end = g_config.input_size + g_config.hidden_size;
    output_start = g_config.input_size + g_config.hidden_size;
    output_end = total_neurons;
    
    std::cout << "[CUDA] Initializing network with " << total_neurons << " neurons..." << std::endl;
    
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
    NetworkCUDAInternal::createNetworkTopology(host_synapses, gen);
    
    total_synapses = host_synapses.size();
    std::cout << "[CUDA] Created " << total_synapses << " synapses" << std::endl;
    
    // Copy synapses to GPU
    safeCudaMalloc(&d_synapses, total_synapses);
    safeCudaMemcpy(d_synapses, host_synapses.data(), total_synapses, cudaMemcpyHostToDevice);
    
    // Initialize random states
    launchRandomStateInit(d_rng_states, total_neurons, rd());
    CUDA_CHECK_KERNEL();
    
    // Zero out buffers
    safeCudaMemset(d_input_buffer, 0, g_config.input_size);
    safeCudaMemset(d_output_buffer, 0, g_config.output_size);
    safeCudaMemset(d_spike_count, 0, 1);
    
    std::cout << "[CUDA] Network initialization complete!" << std::endl;
    g_stats.reset();
}

// Enhanced network topology creation
namespace NetworkCUDAInternal {
void createNetworkTopology(std::vector<GPUSynapse>& synapses, std::mt19937& gen) {
    std::uniform_real_distribution<float> weight_dist(0.0f, g_config.weight_init_std);
    std::uniform_real_distribution<float> delay_dist(g_config.delay_min, g_config.delay_max);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    synapses.clear();
    synapses.reserve(total_neurons * 50); // Conservative estimate
    
    // Input to Hidden connections with cortical organization
    for (int pre = input_start; pre < input_end; ++pre) {
        for (int post = hidden_start; post < hidden_end; ++post) {
            if (prob_dist(gen) < g_config.input_hidden_prob) {
                GPUSynapse syn;
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(gen) * (prob_dist(gen) < g_config.exc_ratio ? 1.0f : -1.0f);
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                synapses.push_back(syn);
            }
        }
    }
    
    // Hidden layer recurrent connections (sparse)
    for (int pre = hidden_start; pre < hidden_end; ++pre) {
        for (int post = hidden_start; post < hidden_end; ++post) {
            if (pre != post && prob_dist(gen) < g_config.hidden_hidden_prob) {
                GPUSynapse syn;
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(gen) * (prob_dist(gen) < g_config.exc_ratio ? 0.5f : -0.8f);
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                synapses.push_back(syn);
            }
        }
    }
    
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
            synapses.push_back(syn);
        }
    }
}

// Enhanced forward pass with better dynamics
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    if (input.size() != g_config.input_size) {
        std::cerr << "[ERROR] Input size mismatch: expected " << g_config.input_size 
                  << ", got " << input.size() << std::endl;
        return std::vector<float>(g_config.output_size, 1.0f / g_config.output_size);
    }
    
    // Copy input to GPU
    safeCudaMemcpy(d_input_buffer, input.data(), g_config.input_size, cudaMemcpyHostToDevice);
    
    // Store reward signal
    safeCudaMemcpy(d_reward_buffer, &reward_signal, 1, cudaMemcpyHostToDevice);
    
    // Calculate simulation steps
    int simulation_steps = static_cast<int>(g_config.simulation_time / g_config.dt);
    
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
        launchRK4NeuronUpdateKernel(d_neurons, total_neurons, g_config.dt);
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

// Improved softmax with numerical stability
std::vector<float> applySoftmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
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

// Network statistics computation
void updateNetworkStatistics() {
    if (total_synapses == 0) return;
    
    // Adaptive STDP parameters based on reward
    float reward_factor = 1.0f + g_config.reward_learning_rate * reward_signal;
    float A_plus = g_config.A_plus * reward_factor;
    float A_minus = g_config.A_minus * (2.0f - reward_factor); // Inverse for depression
    
    // Apply STDP with reward modulation
    launchSTDPUpdateKernel(d_synapses, d_neurons, total_synapses,
                           A_plus, A_minus, g_config.tau_plus, g_config.tau_minus,
                           current_time, g_config.min_weight, g_config.max_weight, 
                           reward_signal);
    CUDA_CHECK_KERNEL();
    
    // Homeostatic mechanisms every 100 updates
    static int update_counter = 0;
    if (++update_counter % 100 == 0 && g_config.homeostatic_strength > 0) {
        NetworkCUDAInternal::applyHomeostaticScaling();
    }
    
    // Update statistics for monitoring
    if (g_config.enable_monitoring && update_counter % g_config.monitoring_interval == 0) {
        NetworkCUDAInternal::updateNetworkStatistics();
    }
    
    g_stats.update_count = update_counter;
    g_stats.reward_signal = reward_signal;
}

// Network statistics computation
void updateNetworkStatistics() {
    // This would compute various network statistics
    // Implementation depends on specific monitoring needs
    static float stats_buffer[4] = {0.0f};
    
    dim3 block = getOptimalBlockSize();
    dim3 grid = getOptimalGridSize(total_neurons);
    computeNetworkStatistics<<<grid, block>>>(
        d_neurons, d_synapses, total_neurons, total_synapses, stats_buffer
    );
    CUDA_CHECK_KERNEL();
    
    // Update global statistics (simplified)
    if (g_stats.update_count % g_config.monitoring_interval == 0) {
        std::cout << "[STATS] Spikes: " << g_stats.total_spikes 
                  << ", Avg Weight: " << g_stats.avg_weight
                  << ", Reward: " << g_stats.reward_signal << std::endl;
    }
}

// Homeostatic scaling to prevent runaway dynamics
void applyHomeostaticScaling() {
    dim3 block = getOptimalBlockSize();
    dim3 grid = getOptimalGridSize(total_synapses);
    
    // Simple homeostatic scaling kernel (would need implementation)
    // This maintains network stability over long training periods
}
} // namespace NetworkCUDAInternal

// Configuration and monitoring functions
void setNetworkConfig(const NetworkConfig& config) {
    g_config = config;
    if (!g_config.validate()) {
        std::cerr << "[ERROR] Invalid network configuration provided" << std::endl;
        throw std::runtime_error("Invalid network configuration");
    }
    std::cout << "[CONFIG] Network configuration updated" << std::endl;
}

NetworkConfig getNetworkConfig() {
    return g_config;
}

void printNetworkStats() {
    std::cout << "\n=== Network Statistics ===" << std::endl;
    std::cout << "Total Neurons: " << total_neurons << std::endl;
    std::cout << "Total Synapses: " << total_synapses << std::endl;
    std::cout << "Input Layer: [" << input_start << ", " << input_end << ")" << std::endl;
    std::cout << "Hidden Layer: [" << hidden_start << ", " << hidden_end << ")" << std::endl;
    std::cout << "Output Layer: [" << output_start << ", " << output_end << ")" << std::endl;
    std::cout << "Current Time: " << current_time << " ms" << std::endl;
    std::cout << "Update Count: " << g_stats.update_count << std::endl;
    std::cout << "Average Firing Rate: " << g_stats.avg_firing_rate << " Hz" << std::endl;
    std::cout << "Average Weight: " << g_stats.avg_weight << std::endl;
    std::cout << "Last Reward Signal: " << g_stats.reward_signal << std::endl;
    std::cout << "=========================" << std::endl;
}

// Advanced features
void saveNetworkState(const std::string& filename) {
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
    
    // Simple binary save (in real implementation, use proper serialization)
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
        std::cerr << "[WARNING] Network size mismatch. Expected: " 
                  << total_neurons << " neurons, " << total_synapses << " synapses. "
                  << "Found: " << loaded_neurons << " neurons, " << loaded_synapses << " synapses." << std::endl;
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

// Additional missing namespace function
namespace NetworkCUDAInternal {
    void validateInputs(const std::vector<float>& input, float reward_signal) {
        if (input.size() != g_config.input_size) {
            throw std::invalid_argument("Input size mismatch");
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
}

// Additional CUDA kernels
__global__ void applyHomeostaticScalingKernel(GPUSynapse* synapses, int num_synapses, 
                                             float scale_factor, float target_rate, float current_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Simple homeostatic scaling based on activity
    float rate_ratio = target_rate / fmaxf(current_rate, 0.001f);
    float scaling = 1.0f + scale_factor * (rate_ratio - 1.0f);
    
    // Apply scaling to synaptic weights
    synapses[idx].weight *= scaling;
    
    // Keep weights within bounds
    synapses[idx].weight = fminf(fmaxf(synapses[idx].weight, -5.0f), 5.0f);
}

__global__ void validateNeuronStates(GPUNeuronState* neurons, int num_neurons, bool* is_valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = neurons[idx];
    
    // Check if neuron state is valid
    bool valid = true;
    valid &= isfinite(neuron.voltage) && neuron.voltage > -150.0f && neuron.voltage < 100.0f;
    valid &= isfinite(neuron.m) && neuron.m >= 0.0f && neuron.m <= 1.0f;
    valid &= isfinite(neuron.h) && neuron.h >= 0.0f && neuron.h <= 1.0f;
    valid &= isfinite(neuron.n) && neuron.n >= 0.0f && neuron.n <= 1.0f;
    
    if (!valid) {
        is_valid[0] = false;
    }
}