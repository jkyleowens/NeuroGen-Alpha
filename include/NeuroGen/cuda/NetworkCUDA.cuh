#pragma once
#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <chrono>

// Include the NetworkConfig definition
#include <NeuroGen/NetworkConfig.h>

// Forward declarations for GPU structures
struct GPUNeuronState;
struct GPUSynapse;
struct GPUCorticalColumn;

// Network statistics structure
struct NetworkStats {
    int total_spikes;
    float average_firing_rate;
    float current_reward;
    float total_simulation_time;
    
    void reset() {
        total_spikes = 0;
        average_firing_rate = 0.0f;
        current_reward = 0.0f;
        total_simulation_time = 0.0f;
    }
};

// Error handling
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

// Constants
namespace NetworkConstants {
    constexpr float DEFAULT_DT = 0.01f;
    constexpr float DEFAULT_SPIKE_THRESHOLD = -40.0f;
    constexpr float DEFAULT_RESTING_POTENTIAL = -65.0f;
    constexpr float MIN_WEIGHT_CONST = -2.0f;
    constexpr float MAX_WEIGHT_CONST = 2.0f;
    constexpr float MIN_DELAY = 0.1f;
    constexpr float MAX_DELAY = 20.0f;
    constexpr int MAX_SIMULATION_STEPS = 10000;
    constexpr int MAX_NEURONS = 100000;
    constexpr int MAX_SYNAPSES = 10000000;
}

// Neuron and synapse type constants
#ifndef SYNAPSE_EXCITATORY
#define SYNAPSE_EXCITATORY 0
#define SYNAPSE_INHIBITORY 1
#endif

#ifndef NEURON_EXCITATORY
#define NEURON_EXCITATORY 0
#define NEURON_INHIBITORY 1
#define NEURON_REWARD_PREDICTION 2
#endif

// Main NetworkCUDA class
class NetworkCUDA {
public:
    // Constructor and destructor
    explicit NetworkCUDA(const NetworkConfig& config);
    ~NetworkCUDA();
    
    // Copy/move constructors (deleted to prevent accidental copying)
    NetworkCUDA(const NetworkCUDA&) = delete;
    NetworkCUDA& operator=(const NetworkCUDA&) = delete;
    NetworkCUDA(NetworkCUDA&&) = delete;
    NetworkCUDA& operator=(NetworkCUDA&&) = delete;
    
    // Core network operations
    void update(float dt_ms, const std::vector<float>& input_currents, float reward);
    std::vector<float> getOutput() const;
    void reset();
    
    // Network state queries
    int getNumNeurons() const { return config.numColumns * config.neuronsPerColumn; }
    int getNumSynapses() const { return static_cast<int>(config.totalSynapses); }
    float getCurrentTime() const { return current_time_ms; }
    NetworkStats getStats() const;
    
    // Network configuration
    void setLearningRate(float rate);
    void setRewardSignal(float reward);
    void enablePlasticity(bool enable);
    
    // Debug and monitoring
    void printNetworkState() const;
    std::vector<float> getNeuronVoltages() const;
    std::vector<float> getSynapticWeights() const;

private:
    // Network configuration
    NetworkConfig config;
    
    // Device memory pointers
    GPUNeuronState* d_neurons;
    GPUSynapse* d_synapses;
    float* d_calcium_levels;
    int* d_neuron_spike_counts;
    curandState* d_random_states;
    GPUCorticalColumn* d_cortical_columns;
    float* d_input_currents;  // Device memory for input data
    
    // Network state
    float current_time_ms;
    bool network_initialized;
    bool plasticity_enabled;
    float current_learning_rate;
    
    // Private methods
    void initializeNetwork();
    void initializeColumns();
    void generateDistanceBasedSynapses();
    void cleanup();
    void allocateDeviceMemory();
    void initializeDeviceArrays();
    void calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const;
    
    // Kernel wrapper methods
    void updateNeuronsWrapper(float dt_ms);
    void updateSynapsesWrapper(float dt_ms);
    void applyPlasticityWrapper(float reward);
    void processSpikingWrapper();
    
    // Validation methods
    void validateConfig() const;
    void checkCudaErrors() const;

    void updateNetworkStatistics();
};

// Global network statistics (managed memory)
extern __managed__ NetworkStats g_stats;

// Utility macros
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw NetworkException(NetworkError::CUDA_ERROR, \
                std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            throw NetworkException(NetworkError::CUDA_ERROR, \
                std::string("CUDA kernel error: ") + cudaGetErrorString(error)); \
        } \
        cudaDeviceSynchronize(); \
    } while(0)

#endif // NETWORK_CUDA_CUH