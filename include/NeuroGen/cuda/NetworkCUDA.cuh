#pragma once
#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/NetworkConfig.h>

// Main interface functions for the neural network
// Core network operations
void initializeNetwork();
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal);
void updateSynapticWeightsCUDA(float reward_signal);
void cleanupNetwork();

// Configuration and monitoring
void setNetworkConfig(const NetworkConfig& config);
NetworkConfig getNetworkConfig();
void printNetworkStats();

// Advanced features
void saveNetworkState(const std::string& filename);
void loadNetworkState(const std::string& filename);
void resetNetwork();

// Internal helper functions (not exposed to main.cpp)
namespace NetworkCUDAInternal {
    void createNetworkTopology(std::vector<GPUSynapse>& synapses,
                               const std::vector<GPUCorticalColumn>& columns,
                               std::mt19937& gen);
    std::vector<float> applySoftmax(const std::vector<float>& input);
    void updateNetworkStatistics();
    void applyHomeostaticScaling();
    void validateInputs(const std::vector<float>& input, float reward_signal);
}

// Forward declaration for GPUNeuronState and GPUSynapse
struct GPUNeuronState;
struct GPUSynapse;

// CUDA kernel declarations for internal use
__global__ void injectInputCurrentImproved(GPUNeuronState* neurons, const float* input_data, 
                                          int input_size, float current_time, float scale);
__global__ void extractOutputImproved(const GPUNeuronState* neurons, float* output_buffer,
                                     int output_size, float current_time);
__global__ void applyRewardModulationImproved(GPUNeuronState* neurons, int num_neurons, float reward);
__global__ void computeNetworkStatistics(const GPUNeuronState* neurons, const GPUSynapse* synapses,
                                        int num_neurons, int num_synapses, float* stats);
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons);
__global__ void applyHomeostaticScalingKernel(GPUSynapse* synapses, int num_synapses, 
                                             float scale_factor, float target_rate, float current_rate);
__global__ void validateNeuronStates(GPUNeuronState* neurons, int num_neurons, bool* is_valid);

// Statistics collected on the GPU network state
struct CudaNetworkStats {
    float avg_firing_rate;
    float total_spikes;
    float avg_weight;
    float reward_signal;
    int   update_count;
};

// Retrieve the latest statistics from the GPU implementation
CudaNetworkStats getNetworkStats();

struct NetworkPerformance {
    float forward_pass_time_ms;
    float learning_time_ms;
    float total_simulation_time_ms;
    int total_decisions_made;
    float average_decision_confidence;
    
    void reset();
    void update(float forward_time, float learning_time);
    void print() const;
};

// Error handling and validation
enum class NetworkError {
    NONE,
    CUDA_ERROR,
    INVALID_INPUT,
    NETWORK_NOT_INITIALIZED,
    MEMORY_ERROR,
    CONFIGURATION_ERROR
};

class NetworkException : public std::exception {
private:
    std::string message_;
    NetworkError error_code_;
    
public:
    NetworkException(NetworkError code, const std::string& message)
        : error_code_(code), message_(message) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    NetworkError getErrorCode() const { return error_code_; }
};

// Utility macros for the network implementation
#define NETWORK_CHECK(condition, error_code, message) \
    do { \
        if (!(condition)) { \
            throw NetworkException(error_code, message); \
        } \
    } while(0)

#define NETWORK_CUDA_CHECK(call, message) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::string full_message = std::string(message) + ": " + cudaGetErrorString(error); \
            throw NetworkException(NetworkError::CUDA_ERROR, full_message); \
        } \
    } while(0)

// Network configuration constants
namespace NetworkConstants {
    constexpr float DEFAULT_DT = 0.01f;                    // 10 μs time step
    constexpr float DEFAULT_SPIKE_THRESHOLD = -40.0f;     // mV
    constexpr float DEFAULT_RESTING_POTENTIAL = -65.0f;   // mV
    constexpr float MIN_WEIGHT = -2.0f;                    // Minimum synaptic weight
    constexpr float MAX_WEIGHT = 2.0f;                     // Maximum synaptic weight
    constexpr float MIN_DELAY = 0.1f;                      // Minimum synaptic delay (ms)
    constexpr float MAX_DELAY = 20.0f;                     // Maximum synaptic delay (ms)
    constexpr int MAX_SIMULATION_STEPS = 10000;           // Safety limit
    constexpr int MAX_NEURONS = 100000;                   // Memory safety limit
    constexpr int MAX_SYNAPSES = 10000000;                // Memory safety limit
}

// Performance profiling utilities
class NetworkProfiler {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<float> timing_data_;
    bool is_profiling_;
    
public:
    NetworkProfiler() : is_profiling_(false) {}
    
    void startProfiling();
    void endProfiling();
    void recordTiming(const std::string& operation_name);
    void printReport() const;
    void reset();
    
    bool isProfiling() const { return is_profiling_; }
};

#endif // NETWORK_CUDA_CUH
