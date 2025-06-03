#ifndef NEUROGENALPHA_NETWORKCONFIG_H
#define NEUROGENALPHA_NETWORKCONFIG_H

#include <string>
#include <cstddef>

/**
 * @brief Network configuration parameters
 */
struct NetworkConfig {
    // Simulation parameters
    double dt = 0.01;                    // Integration time step (ms)
    double axonal_speed = 1.0;           // m/s (affects delays)
    
    // Spatial organization
    double network_width = 1000.0;       // μm
    double network_height = 1000.0;      // μm
    double network_depth = 100.0;        // μm
    
    // Connectivity parameters
    double max_connection_distance = 200.0; // μm
    double connection_probability_base = 0.01;
    double distance_decay_constant = 50.0;   // μm
    double spike_correlation_window = 20.0;  // ms
    double correlation_threshold = 0.3;
    
    // Neurogenesis parameters
    bool enable_neurogenesis = true;
    double neurogenesis_rate = 0.001;        // neurons/ms base rate
    double activity_threshold_low = 0.1;     // For underactivation
    double activity_threshold_high = 10.0;   // For hyperactivation
    size_t max_neurons = 1000;
    
    // Pruning parameters
    bool enable_pruning = true;
    double synapse_pruning_threshold = 0.05;
    double neuron_pruning_threshold = 0.01;
    double pruning_check_interval = 100.0;   // ms
    double synapse_activity_window = 1000.0; // ms
    
    // Plasticity parameters
    bool enable_stdp = true;
    double stdp_learning_rate = 0.01;
    double stdp_tau_pre = 20.0;              // ms
    double stdp_tau_post = 20.0;             // ms
    double eligibility_decay = 50.0;         // ms
    double min_synaptic_weight = 0.001;      // Minimum synaptic weight
    double max_synaptic_weight = 2.0;        // Maximum synaptic weight
    
    // Neuromodulation
    bool enable_neuromodulation = true;
    double modulation_strength = 0.1;

    // Spike threshold
    double spike_threshold = 30.0;           // mV

    std::string toString() const {
        return "NetworkConfig{dt=" + std::to_string(dt) +
               ", max_neurons=" + std::to_string(max_neurons) + "}";
    }
};

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

#endif // NEUROGENALPHA_NETWORKCONFIG_H