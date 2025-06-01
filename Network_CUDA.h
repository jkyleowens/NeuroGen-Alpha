/**
 * @file Network_CUDA.h
 * @brief CUDA-Enhanced Biologically Inspired Dynamic Neural Network
 * 
 * This header extends the original Network.h with GPU acceleration capabilities
 * while maintaining full backward compatibility with CPU-only operation.
 * 
 * @author Neural Dynamics Lab
 * @version 2.1 - CUDA Enhanced
 */

#ifndef NETWORK_CUDA_H
#define NETWORK_CUDA_H

#include "Network.h"

// Conditional CUDA includes
#ifdef USE_CUDA
#include "Network.cuh"
#include <cuda_runtime.h>
#include <memory>
#endif

/**
 * @brief Enhanced Network configuration with CUDA options
 */
struct NetworkConfigCUDA : public NetworkConfig {
    // CUDA-specific parameters
    bool enable_cuda = false;           // Enable GPU acceleration
    bool force_gpu_sync = false;        // Force synchronization after each kernel
    int cuda_device_id = 0;             // CUDA device to use
    size_t gpu_memory_limit = 0;        // GPU memory limit (0 = auto)
    
    // Performance tuning
    int threads_per_block = 256;        // CUDA threads per block
    bool use_pinned_memory = true;      // Use pinned host memory for faster transfers
    bool async_memory_transfer = true;  // Async memory transfers
    
    // Hybrid CPU-GPU processing
    float gpu_load_threshold = 100;     // Minimum neurons to use GPU
    bool adaptive_processing = true;    // Automatically choose CPU vs GPU
    
    std::string toString() const {
        return NetworkConfig::toString() + 
               ", cuda_enabled=" + (enable_cuda ? "true" : "false") +
               ", device_id=" + std::to_string(cuda_device_id);
    }
};

/**
 * @brief CUDA-enhanced Network class
 * 
 * Note: This class inherits from Network but the base class methods are not virtual.
 * This means NetworkCUDA should be used directly rather than through Network* pointers
 * for CUDA functionality to be active. For polymorphic usage, use the factory functions
 * createOptimalNetwork() or createNetwork() which return the appropriate type.
 */
class NetworkCUDA : public Network {
private:
#ifdef USE_CUDA
    // CUDA components
    std::unique_ptr<CUDANetworkProcessor> cuda_processor_;
    std::unique_ptr<CUDAMemoryManager> cuda_memory_;
    
    // GPU state management
    std::vector<GPUNeuronState> gpu_neurons_;
    std::vector<GPUSynapse> gpu_synapses_;
    
    // Performance tracking
    mutable float total_gpu_time_;
    mutable float total_cpu_time_;
    mutable int gpu_kernel_calls_;
    mutable int cpu_step_calls_;
    
    // Memory management
    bool data_on_gpu_;
    bool gpu_initialized_;
    cudaStream_t main_stream_;
    cudaStream_t memory_stream_;
    
    // Hybrid processing state
    mutable bool use_gpu_this_step_;
    size_t gpu_switch_threshold_;
#endif

    NetworkConfigCUDA cuda_config_;
    bool cuda_available_;
    bool cuda_enabled_;

public:
    /**
     * @brief Construct CUDA-enhanced network
     */
    explicit NetworkCUDA(const NetworkConfigCUDA& config = NetworkConfigCUDA());
    
    /**
     * @brief Destructor with CUDA cleanup
     */
    virtual ~NetworkCUDA();
    
    // === CUDA-specific Methods ===
    
    /**
     * @brief Initialize CUDA subsystem
     */
    bool initializeCUDA();
    
    /**
     * @brief Check if CUDA is available and initialized
     */
    bool isCUDAEnabled() const { return cuda_enabled_; }
    
    /**
     * @brief Get CUDA device information
     */
    std::string getCUDADeviceInfo() const;
    
    /**
     * @brief Set CUDA device
     */
    bool setCUDADevice(int device_id);
    
    /**
     * @brief Force data synchronization between CPU and GPU
     */
    void synchronizeGPUData();
    
    /**
     * @brief Transfer data to GPU (if not already there)
     */
    void uploadToGPU();
    
    /**
     * @brief Transfer data from GPU to CPU
     */
    void downloadFromGPU();
    
    /**
     * @brief Enable/disable GPU processing
     */
    void setGPUEnabled(bool enabled);
    
    /**
     * @brief Get performance statistics
     */
    struct CUDAPerformanceStats {
        float total_gpu_time;
        float total_cpu_time;
        int gpu_kernel_calls;
        int cpu_step_calls;
        float gpu_speedup;
        float memory_transfer_time;
        size_t gpu_memory_used;
        size_t cpu_memory_used;
    };
    
    CUDAPerformanceStats getPerformanceStats() const;
    void resetPerformanceCounters();
    
    // === Overridden Network Methods with CUDA acceleration ===
    
    /**
     * @brief CUDA-accelerated simulation step
     */
    void step(double dt);
    
    /**
     * @brief CUDA-accelerated network run
     */
    void run(double duration, double dt_sim);
    
    /**
     * @brief CUDA-accelerated plasticity update
     */
    void updatePlasticity();
    
    /**
     * @brief CUDA-accelerated synaptogenesis
     */
    void performSynaptogenesis();
    
    /**
     * @brief CUDA-accelerated neurogenesis
     */
    void performNeurogenesis();
    
    /**
     * @brief CUDA-accelerated pruning
     */
    void performPruning();
    
    /**
     * @brief CUDA-accelerated current injection
     */
    void injectCurrent(size_t neuron_id, double current);
    
    /**
     * @brief CUDA-accelerated neuromodulator release
     */
    void releaseNeuromodulator(const std::string& type, double amount);
    
    /**
     * @brief CUDA-accelerated statistics calculation
     */
    NetworkStats calculateNetworkStats(double time_window = 1000.0) const;
    
    /**
     * @brief CUDA-accelerated regional activity calculation
     */
    std::vector<double> getRegionalActivity(size_t x_bins = 10, size_t y_bins = 10, 
                                          double time_window = 100.0) const;
    
    /**
     * @brief Perform a simulation step
     */
    void step(float dt, float threshold);

    /**
     * @brief Get the spike count from the GPU
     */
    int getDeviceSpikeCount() const;
    
    // === Memory Management ===
    
    /**
     * @brief Estimate GPU memory requirements
     */
    size_t estimateGPUMemoryUsage() const;
    
    /**
     * @brief Check available GPU memory
     */
    size_t getAvailableGPUMemory() const;
    
    /**
     * @brief Optimize memory layout for GPU processing
     */
    void optimizeMemoryLayout();
    
    // === Configuration ===
    
    /**
     * @brief Update CUDA configuration
     */
    void updateCUDAConfig(const NetworkConfigCUDA& config);
    
    /**
     * @brief Get current CUDA configuration
     */
    const NetworkConfigCUDA& getCUDAConfig() const { return cuda_config_; }

private:
    // === Private CUDA Methods ===
    
#ifdef USE_CUDA
    /**
     * @brief Convert CPU neuron data to GPU format
     */
    GPUNeuronState convertNeuronToGPU(size_t neuron_id);
    
    /**
     * @brief Convert CPU synapse data to GPU format
     */
    GPUSynapse convertSynapseToGPU(size_t synapse_id);
    
    /**
     * @brief Convert all neurons to GPU format
     */
    void convertNeuronsToGPU();
    
    /**
     * @brief Convert all synapses to GPU format  
     */
    void convertSynapsesToGPU();
    
    /**
     * @brief Convert GPU data back to CPU format
     */
    void convertDataFromGPU();
    
    /**
     * @brief Decide whether to use GPU for current operation
     */
    bool shouldUseGPU() const;
    
    /**
     * @brief Cleanup CUDA resources
     */
    void cleanupCUDA();
    
    /**
     * @brief Handle CUDA errors gracefully
     */
    void handleCUDAError(const std::string& operation);
    
    /**
     * @brief Benchmark CPU vs GPU performance
     */
    void benchmarkProcessingModes();
#endif

    /**
     * @brief Fallback to CPU processing
     */
    void fallbackToCPU();
    
    /**
     * @brief Check CUDA availability at runtime
     */
    bool checkCUDAAvailability();
};

/**
 * @brief CUDA-enhanced NetworkBuilder
 */
class NetworkBuilderCUDA {
private:
    NetworkConfigCUDA cuda_config_;
    
    // Copy needed members from base NetworkBuilder
    struct NeuronPopulation {
        NeuronFactory::NeuronType type;
        size_t count;
        Position3D position;
        double radius;
    };
    
    std::vector<NeuronPopulation> neuronPopulations_;
    double connectionProbability_;
    
    Position3D randomizePosition(const Position3D& center, double radius);
    
public:
    NetworkBuilderCUDA();
    
    /**
     * @brief Set CUDA configuration
     */
    NetworkBuilderCUDA& setCUDAConfig(const NetworkConfigCUDA& config);
    
    /**
     * @brief Add neuron population  
     */
    NetworkBuilderCUDA& addNeuronPopulation(NeuronFactory::NeuronType type,
                                           size_t count,
                                           const Position3D& position,
                                           double radius);
    
    /**
     * @brief Add random connections
     */
    NetworkBuilderCUDA& addRandomConnections(double probability);
    
    /**
     * @brief Enable CUDA acceleration
     */
    NetworkBuilderCUDA& enableCUDA(bool enable = true);
    
    /**
     * @brief Set CUDA device
     */
    NetworkBuilderCUDA& setCUDADevice(int device_id);
    
    /**
     * @brief Set GPU memory limit
     */
    NetworkBuilderCUDA& setGPUMemoryLimit(size_t limit_bytes);
    
    /**
     * @brief Build CUDA-enhanced network
     */
    std::shared_ptr<NetworkCUDA> buildCUDA();
    
    /**
     * @brief Build with automatic CUDA detection
     */
    std::shared_ptr<NetworkCUDA> buildAuto();
};

/**
 * @brief CUDA utility functions
 */
namespace NetworkCUDAUtils {
    
    /**
     * @brief Check if CUDA is available on system
     */
    bool isCUDAAvailable();
    
    /**
     * @brief Get number of CUDA devices
     */
    int getCUDADeviceCount();
    
    /**
     * @brief Get CUDA device properties
     */
    std::string getCUDADeviceInfo(int device_id);
    
    /**
     * @brief Get optimal CUDA configuration for given network size
     */
    NetworkConfigCUDA getOptimalCUDAConfig(size_t num_neurons, size_t num_synapses);
    
    /**
     * @brief Benchmark CUDA vs CPU performance
     */
    struct BenchmarkResult {
        double cpu_time_ms;
        double gpu_time_ms;
        double speedup_factor;
        bool cuda_faster;
    };
    
    BenchmarkResult benchmarkCUDAPerformance(size_t num_neurons, size_t num_synapses, 
                                           double simulation_time = 100.0);
    
    /**
     * @brief Recommend CUDA settings based on hardware
     */
    NetworkConfigCUDA recommendCUDASettings();
    
    /**
     * @brief Memory optimization utilities
     */
    size_t calculateOptimalBatchSize(size_t available_memory, size_t num_neurons);
    bool canFitOnGPU(size_t num_neurons, size_t num_synapses);
    
    /**
     * @brief Error handling
     */
    std::string getLastCUDAError();
    void clearCUDAErrors();
}

// === Preprocessor Macros for CUDA Compilation ===

#ifdef USE_CUDA
    #define NETWORK_CUDA_ENABLED 1
    #define NETWORK_CLASS NetworkCUDA
    #define NETWORK_BUILDER NetworkBuilderCUDA
    #define NETWORK_CONFIG NetworkConfigCUDA
#else
    #define NETWORK_CUDA_ENABLED 0
    #define NETWORK_CLASS Network
    #define NETWORK_BUILDER NetworkBuilder  
    #define NETWORK_CONFIG NetworkConfig
#endif

// === Type Aliases for Easy Switching ===

#ifdef USE_CUDA
using NetworkType = NetworkCUDA;
using NetworkBuilderType = NetworkBuilderCUDA;
using NetworkConfigType = NetworkConfigCUDA;
#else
using NetworkType = Network;
using NetworkBuilderType = NetworkBuilder;
using NetworkConfigType = NetworkConfig;
#endif

/**
 * @brief Factory function for creating networks with automatic CUDA detection
 * 
 * This function returns a NetworkCUDA if CUDA is available, otherwise a regular Network.
 * The returned pointer should be used directly (not cast to base class) to access
 * CUDA-specific functionality.
 */
std::shared_ptr<NetworkType> createOptimalNetwork(const NetworkConfigType& config = NetworkConfigType());

/**
 * @brief Factory function for creating networks with explicit CUDA preference
 * 
 * @param config Network configuration
 * @param prefer_cuda Whether to prefer CUDA if available
 * @return Shared pointer to network (NetworkCUDA if CUDA used, Network otherwise)
 */
std::shared_ptr<NetworkType> createNetwork(const NetworkConfigType& config, bool prefer_cuda = true);

#endif // NETWORK_CUDA_H