#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "STDPKernel.cuh"
#include <curand_kernel.h>
#include "NetworkCUDA.cuh"
#include "GPUNeuralStructures.h"
#include "STDPKernel.cuh"

// Network topology and GPU state
static GPUNeuronState* d_neurons = nullptr;
static GPUSynapse* d_synapses = nullptr;
static float* d_input_buffer = nullptr;
static float* d_output_buffer = nullptr;
static GPUSpikeEvent* d_spike_events = nullptr;
static int* d_spike_count = nullptr;
static curandState* d_rng_states = nullptr;

// Network dimensions
static constexpr int INPUT_SIZE = 60;     // Feature vector size from main.cpp
static constexpr int HIDDEN_SIZE = 512;   // Hidden layer neurons
static constexpr int OUTPUT_SIZE = 3;     // buy/sell/hold decisions
static constexpr int TOTAL_NEURONS = INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
static constexpr int MAX_SYNAPSES = HIDDEN_SIZE * (INPUT_SIZE + OUTPUT_SIZE) + (HIDDEN_SIZE * HIDDEN_SIZE / 4);

static int num_neurons = TOTAL_NEURONS;
static int num_synapses = 0; // Will be set during initialization

// Network layer boundaries
static constexpr int INPUT_START = 0;
static constexpr int INPUT_END = INPUT_SIZE;
static constexpr int HIDDEN_START = INPUT_SIZE;
static constexpr int HIDDEN_END = INPUT_SIZE + HIDDEN_SIZE;
static constexpr int OUTPUT_START = INPUT_SIZE + HIDDEN_SIZE;
static constexpr int OUTPUT_END = TOTAL_NEURONS;

// Training parameters
static float current_time = 0.0f;
static constexpr float dt = 0.01f;
static constexpr float spike_threshold = -40.0f;

// Initialize the CUDA-powered neural network
void initializeNetwork() {
    std::cout << "[CUDA] Initializing neural network..." << std::endl;
    
    // Allocate GPU memory for neurons
    cudaMalloc(&d_neurons, num_neurons * sizeof(GPUNeuronState));
    cudaMemset(d_neurons, 0, num_neurons * sizeof(GPUNeuronState));
    
    // Initialize neurons with proper resting state
    std::vector<GPUNeuronState> host_neurons(num_neurons);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> voltage_dist(-65.0f, 2.0f);
    std::uniform_real_distribution<float> gating_dist(0.0f, 1.0f);
    
    for (int i = 0; i < num_neurons; ++i) {
        host_neurons[i].voltage = voltage_dist(gen);
        host_neurons[i].spiked = false;
        host_neurons[i].last_spike_time = -1000.0f; // Long ago
        
        // Initialize HH gating variables with small random perturbations
        host_neurons[i].m = 0.05f + gating_dist(gen) * 0.02f;
        host_neurons[i].h = 0.60f + gating_dist(gen) * 0.02f;
        host_neurons[i].n = 0.32f + gating_dist(gen) * 0.02f;
        
        host_neurons[i].compartment_count = 1;
        host_neurons[i].voltages[0] = host_neurons[i].voltage;
        host_neurons[i].I_leak[0] = 0.0f;
        host_neurons[i].Cm[0] = 1.0f;
    }
    
    cudaMemcpy(d_neurons, host_neurons.data(), num_neurons * sizeof(GPUNeuronState), cudaMemcpyHostToDevice);
    
    // Create network topology: Input -> Hidden -> Output
    std::vector<GPUSynapse> host_synapses;
    std::uniform_real_distribution<float> weight_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> delay_dist(0.5f, 2.0f);
    
    // Input to Hidden connections (fully connected)
    for (int pre = INPUT_START; pre < INPUT_END; ++pre) {
        for (int post = HIDDEN_START; post < HIDDEN_END; ++post) {
            if (gen() % 100 < 80) { // 80% connection probability
                GPUSynapse syn;
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(gen) * 0.5f; // Smaller initial weights
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                host_synapses.push_back(syn);
            }
        }
    }
    
    // Hidden to Hidden connections (sparse recurrent)
    for (int pre = HIDDEN_START; pre < HIDDEN_END; ++pre) {
        for (int post = HIDDEN_START; post < HIDDEN_END; ++post) {
            if (pre != post && gen() % 100 < 10) { // 10% recurrent connectivity
                GPUSynapse syn;
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(gen) * 0.3f;
                syn.delay = delay_dist(gen);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                host_synapses.push_back(syn);
            }
        }
    }
    
    // Hidden to Output connections (fully connected)
    for (int pre = HIDDEN_START; pre < HIDDEN_END; ++pre) {
        for (int post = OUTPUT_START; post < OUTPUT_END; ++post) {
            GPUSynapse syn;
            syn.pre_neuron_idx = pre;
            syn.post_neuron_idx = post;
            syn.weight = weight_dist(gen) * 0.2f;
            syn.delay = delay_dist(gen);
            syn.last_pre_spike_time = -1000.0f;
            syn.activity_metric = 0.0f;
            host_synapses.push_back(syn);
        }
    }
    
    num_synapses = host_synapses.size();
    std::cout << "[CUDA] Created " << num_synapses << " synapses" << std::endl;
    
    // Allocate and copy synapses to GPU
    cudaMalloc(&d_synapses, num_synapses * sizeof(GPUSynapse));
    cudaMemcpy(d_synapses, host_synapses.data(), num_synapses * sizeof(GPUSynapse), cudaMemcpyHostToDevice);
    
    // Allocate working buffers
    cudaMalloc(&d_input_buffer, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_buffer, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_spike_events, num_neurons * sizeof(GPUSpikeEvent));
    cudaMalloc(&d_spike_count, sizeof(int));
    cudaMalloc(&d_rng_states, num_neurons * sizeof(curandState));
    
    // Initialize random states for neurons
    launchRandomStateInit(d_rng_states, num_neurons, rd());
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "[CUDA ERROR] Network initialization failed: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "[CUDA] Network initialized successfully with:" << std::endl;
        std::cout << "  - " << num_neurons << " neurons (" << INPUT_SIZE << " input, " 
                  << HIDDEN_SIZE << " hidden, " << OUTPUT_SIZE << " output)" << std::endl;
        std::cout << "  - " << num_synapses << " synapses" << std::endl;
    }
}

// Perform a forward pass through the CUDA network
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    if (input.size() != INPUT_SIZE) {
        std::cerr << "[CUDA ERROR] Input size mismatch: expected " << INPUT_SIZE 
                  << ", got " << input.size() << std::endl;
        return std::vector<float>(OUTPUT_SIZE, 0.0f);
    }
    
    // Copy input to GPU and inject into input neurons
    cudaMemcpy(d_input_buffer, input.data(), INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Inject input as current to input neurons (simple rate coding)
    // We'll need a kernel to convert input features to neural currents
    injectInputCurrent<<<(INPUT_SIZE + 255) / 256, 256>>>(
        d_neurons + INPUT_START, d_input_buffer, INPUT_SIZE, current_time
    );
    
    // Run neural dynamics for several time steps to process input
    constexpr int PROCESSING_STEPS = 10; // Process input over 10ms
    
    for (int step = 0; step < PROCESSING_STEPS; ++step) {
        current_time += dt;
        
        // Update neuron membrane dynamics using RK4
        launchRK4NeuronUpdateKernel(d_neurons, num_neurons, dt);
        
        // Detect spikes
        cudaMemset(d_spike_count, 0, sizeof(int));
        launchSpikeDetectionKernel(d_neurons, d_spike_events, spike_threshold, 
                                   d_spike_count, num_neurons, current_time);
        
        // Propagate spikes through synapses
        launchSynapseInputKernel(d_synapses, d_neurons, num_synapses);
        
        // Apply reward modulation to dopaminergic influence
        if (step == PROCESSING_STEPS - 1) { // Apply reward at end of processing
            applyRewardModulation<<<(num_neurons + 255) / 256, 256>>>(
                d_neurons, num_neurons, reward_signal
            );
        }
    }
    
    // Extract output from output neurons
    std::vector<float> raw_output(OUTPUT_SIZE);
    extractNeuralOutput<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        d_neurons + OUTPUT_START, d_output_buffer, OUTPUT_SIZE, current_time
    );
    cudaMemcpy(raw_output.data(), d_output_buffer, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Apply softmax normalization for decision probabilities
    std::vector<float> output(OUTPUT_SIZE);
    float sum_exp = 0.0f;
    float max_val = *std::max_element(raw_output.begin(), raw_output.end());
    
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = expf(raw_output[i] - max_val); // Subtract max for numerical stability
        sum_exp += output[i];
    }
    
    if (sum_exp > 0.0f) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            output[i] /= sum_exp;
        }
    } else {
        // Fallback to uniform distribution
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            output[i] = 1.0f / OUTPUT_SIZE;
        }
    }
    
    return output;
}

// Update synaptic weights using STDP on the GPU
void updateSynapticWeightsCUDA(float reward_signal) {
    // STDP parameters tuned for financial trading
    float A_plus = 0.008f * (1.0f + reward_signal * 0.1f);   // Reward modulates learning rate
    float A_minus = 0.010f * (1.0f - reward_signal * 0.05f); // Asymmetric modulation
    float tau_plus = 20.0f;
    float tau_minus = 25.0f;
    float w_min = -1.0f;     // Allow negative weights (inhibitory)
    float w_max = 1.0f;
    
    // Apply reward-modulated STDP
    launchSTDPUpdateKernel(d_synapses, d_neurons, num_synapses,
                           A_plus, A_minus, tau_plus, tau_minus,
                           current_time, w_min, w_max, reward_signal);
    
    // Apply homeostatic scaling every 100 updates to prevent runaway dynamics
    static int update_counter = 0;
    if (++update_counter % 100 == 0) {
        applyHomeostaticScaling<<<(num_synapses + 255) / 256, 256>>>(
            d_synapses, num_synapses, 0.99f // Slight scaling factor
        );
    }
    
    // Prune very weak synapses occasionally for efficiency
    if (update_counter % 1000 == 0) {
        pruneSynapses<<<(num_synapses + 255) / 256, 256>>>(
            d_synapses, num_synapses, 0.001f // Minimum weight threshold
        );
    }
}

// Cleanup function to free GPU memory
void cleanupNetwork() {
    if (d_neurons) cudaFree(d_neurons);
    if (d_synapses) cudaFree(d_synapses);
    if (d_input_buffer) cudaFree(d_input_buffer);
    if (d_output_buffer) cudaFree(d_output_buffer);
    if (d_spike_events) cudaFree(d_spike_events);
    if (d_spike_count) cudaFree(d_spike_count);
    if (d_rng_states) cudaFree(d_rng_states);
    
    d_neurons = nullptr;
    d_synapses = nullptr;
    d_input_buffer = nullptr;
    d_output_buffer = nullptr;
    d_spike_events = nullptr;
    d_spike_count = nullptr;
    d_rng_states = nullptr;
}

// Helper CUDA kernels for network operations
__global__ void injectInputCurrent(GPUNeuronState* input_neurons, float* input_data, 
                                  int input_size, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        // Convert input features to membrane current injection
        // Scale features to reasonable current range (0-50 pA)
        float current = input_data[idx] * 20.0f + 10.0f; // Bias towards depolarization
        
        // Inject current by directly modifying voltage
        input_neurons[idx].voltage += current * 0.01f; // Small integration step
        
        // Ensure voltage stays in reasonable range
        input_neurons[idx].voltage = fminf(fmaxf(input_neurons[idx].voltage, -80.0f), -40.0f);
    }
}

__global__ void extractNeuralOutput(GPUNeuronState* output_neurons, float* output_buffer,
                                   int output_size, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        // Convert membrane potential to output signal
        // Higher voltage = stronger signal
        float voltage = output_neurons[idx].voltage;
        
        // Scale voltage to output range with sigmoid-like function
        output_buffer[idx] = 1.0f / (1.0f + expf(-(voltage + 55.0f) / 10.0f));
        
        // Add spike history contribution
        float time_since_spike = current_time - output_neurons[idx].last_spike_time;
        if (time_since_spike < 50.0f) { // Within 50ms of last spike
            output_buffer[idx] += expf(-time_since_spike / 20.0f) * 0.5f;
        }
        
        // Clamp output
        output_buffer[idx] = fminf(fmaxf(output_buffer[idx], 0.0f), 2.0f);
    }
}

__global__ void applyRewardModulation(GPUNeuronState* neurons, int num_neurons, float reward) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        // Reward modulates excitability through leak current adjustment
        float modulation = reward * 0.1f; // Scale reward signal
        
        // Positive reward increases excitability, negative decreases it
        neurons[idx].I_leak[0] += modulation;
        
        // Keep leak current in reasonable bounds
        neurons[idx].I_leak[0] = fminf(fmaxf(neurons[idx].I_leak[0], -5.0f), 5.0f);
    }
}

__global__ void applyHomeostaticScaling(GPUSynapse* synapses, int num_synapses, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_synapses) {
        synapses[idx].weight *= scale_factor;
    }
}

__global__ void pruneSynapses(GPUSynapse* synapses, int num_synapses, float min_weight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_synapses) {
        if (fabsf(synapses[idx].weight) < min_weight) {
            synapses[idx].weight = 0.0f; // Effectively prune by setting to zero
        }
    }
}
