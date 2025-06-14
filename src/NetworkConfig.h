#ifndef NEUROGENALPHA_NETWORKCONFIG_H
#define NEUROGENALPHA_NETWORKCONFIG_H

#include <iostream>
#include <string>
#include <cstddef>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#endif

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
    size_t max_neurons = 4096;
    
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
    
    // STDP parameters (additional for CUDA compatibility)
    float reward_learning_rate = 0.01f;      // Reward modulation learning rate
    float A_plus = 0.01f;                    // STDP potentiation amplitude
    float A_minus = 0.012f;                  // STDP depression amplitude
    float tau_plus = 20.0f;                  // STDP potentiation time constant (ms)
    float tau_minus = 20.0f;                 // STDP depression time constant (ms)
    float min_weight = 0.001f;               // Minimum synaptic weight (alias for CUDA)
    float max_weight = 2.0f;                 // Maximum synaptic weight (alias for CUDA)
    
    // Homeostatic parameters
    float homeostatic_strength = 0.001f;     // Homeostatic scaling strength
    
    // Network topology parameters
    int input_size = 64;                     // Number of input neurons
    int output_size = 10;                    // Number of output neurons
    int hidden_size = 256;                   // Number of hidden neurons
    
    // Connection probabilities
    float input_hidden_prob = 0.8f;          // Input to hidden connection probability
    float hidden_hidden_prob = 0.1f;         // Hidden to hidden connection probability
    float hidden_output_prob = 0.9f;         // Hidden to output connection probability
    
    // Weight initialization
    float weight_init_std = 0.5f;            // Standard deviation for weight initialization
    float delay_min = 1.0f;                  // Minimum synaptic delay (ms)
    float delay_max = 5.0f;                  // Maximum synaptic delay (ms)
    
    // Input parameters
    float input_current_scale = 10.0f;       // Scale factor for input current injection    

    float exc_ratio = 0.8f;  // Excitatory connection ratio
    float simulation_time = 50.0f;

    // TopologyGenerator-specific fields
    int numColumns = 4;                       // Number of cortical columns
    int neuronsPerColumn = 256;               // Neurons per column
    int localFanOut = 30;                     // Local fan-out connections per neuron
    int localFanIn = 30;                      // Local fan-in connections per neuron
    
    // Synaptic weight ranges
    float wExcMin = 0.05f;                    // Minimum excitatory weight
    float wExcMax = 0.15f;                    // Maximum excitatory weight
    float wInhMin = 0.20f;                    // Minimum inhibitory weight  
    float wInhMax = 0.40f;                    // Maximum inhibitory weight
    
    // Synaptic delay ranges
    float dMin = 0.5f;                        // Minimum synaptic delay (ms)
    float dMax = 2.0f;                        // Maximum synaptic delay (ms)
    
    // Computed fields
    size_t totalSynapses = 0;                 // Total synapses (computed by finalizeConfig)

    bool enable_monitoring = true;
    int monitoring_interval = 100;
    
    // Neuromodulation
    bool enable_neuromodulation = true;
    double modulation_strength = 0.1;

    // Spike threshold
    double spike_threshold = 30.0;           // mV
    
    // Add missing methods:
    void print() const {
        std::cout << "=== Network Configuration ===" << std::endl;
        std::cout << "Input Size: " << input_size << std::endl;
        std::cout << "Hidden Size: " << hidden_size << std::endl;
        std::cout << "Output Size: " << output_size << std::endl;
        std::cout << "Simulation Time: " << simulation_time << " ms" << std::endl;
        std::cout << "Time Step: " << dt << " ms" << std::endl;
        std::cout << "Excitatory Ratio: " << exc_ratio << std::endl;
        std::cout << "============================" << std::endl;
    }
    
    // Validation method
    bool validate() const {
        return input_size > 0 && output_size > 0 && hidden_size > 0 &&
               min_weight >= 0.0f && max_weight > min_weight &&
               tau_plus > 0.0f && tau_minus > 0.0f &&
               A_plus >= 0.0f && A_minus >= 0.0f &&
               numColumns > 0 && neuronsPerColumn > 0 &&
               localFanOut > 0 && wExcMin >= 0.0f && wExcMax > wExcMin &&
               wInhMin >= 0.0f && wInhMax > wInhMin &&
               dMin > 0.0f && dMax > dMin;
    }
    
    // Finalize configuration by computing derived values
    void finalizeConfig() {
        // Compute total synapses based on topology parameters
        totalSynapses = static_cast<size_t>(numColumns) * 
                       static_cast<size_t>(neuronsPerColumn) * 
                       static_cast<size_t>(localFanOut);
        
        // Ensure weight ranges are consistent
        if (wExcMax <= wExcMin) {
            wExcMax = wExcMin + 0.1f;
        }
        if (wInhMax <= wInhMin) {
            wInhMax = wInhMin + 0.1f;
        }
        if (dMax <= dMin) {
            dMax = dMin + 0.5f;
        }
        
        // Update total network size estimates
        hidden_size = numColumns * neuronsPerColumn;
        
        // Ensure minimum time step for stability
        if (dt <= 0.0) {
            dt = 0.01; // Default 0.01ms time step
        }
    }
    
    std::string toString() const {
        return "NetworkConfig{dt=" + std::to_string(dt) +
               ", max_neurons=" + std::to_string(hidden_size) + 
               ", numColumns=" + std::to_string(numColumns) +
               ", neuronsPerColumn=" + std::to_string(neuronsPerColumn) + "}";
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
