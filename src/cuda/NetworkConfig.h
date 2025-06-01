#pragma once
#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <string>
#include <iostream>

struct NetworkConfig {
    // Network topology
    int input_size = 60;           // Feature vector size
    int hidden_size = 512;         // Hidden layer neurons  
    int output_size = 3;           // buy/sell/hold decisions
    
    // Cortical column structure
    int num_columns = 8;           // Number of cortical columns
    int neurons_per_column = 128;  // Neurons per column
    float exc_ratio = 0.8f;        // 80% excitatory neurons
    
    // Connection probabilities
    float input_hidden_prob = 0.8f;    // Input to hidden connectivity
    float hidden_hidden_prob = 0.1f;   // Recurrent connectivity
    float hidden_output_prob = 1.0f;   // Hidden to output (fully connected)
    float local_connectivity = 0.3f;   // Within-column connectivity
    float inter_column_prob = 0.05f;   // Between-column connectivity
    
    // Simulation parameters
    float dt = 0.01f;              // Integration time step (ms)
    float simulation_time = 100.0f; // Time to run for each decision (ms)
    float spike_threshold = -40.0f; // Spike detection threshold (mV)
    
    // Synaptic parameters
    float weight_init_std = 0.1f;   // Initial weight standard deviation
    float max_weight = 1.0f;        // Maximum synaptic weight
    float min_weight = -1.0f;       // Minimum synaptic weight (allows inhibition)
    float delay_min = 0.5f;         // Minimum synaptic delay (ms)
    float delay_max = 5.0f;         // Maximum synaptic delay (ms)
    
    // STDP parameters
    float A_plus = 0.01f;          // Potentiation amplitude
    float A_minus = 0.012f;        // Depression amplitude  
    float tau_plus = 20.0f;        // Potentiation time constant (ms)
    float tau_minus = 25.0f;       // Depression time constant (ms)
    float stdp_window = 100.0f;    // STDP time window (ms)
    
    // Reward modulation
    float reward_learning_rate = 0.1f;   // How much reward affects learning
    float baseline_reward = 0.0f;        // Baseline reward level
    float reward_decay = 0.95f;          // Reward signal decay
    
    // Homeostasis
    float homeostatic_strength = 0.001f; // Homeostatic scaling strength
    float target_firing_rate = 5.0f;     // Target firing rate (Hz)
    float activity_decay = 0.99f;        // Activity trace decay
    
    // Input encoding
    float input_current_scale = 20.0f;   // Scale factor for input currents
    float input_noise_std = 2.0f;        // Input noise standard deviation
    
    // Performance monitoring
    bool enable_monitoring = true;       // Enable performance monitoring
    int monitoring_interval = 1000;     // Monitoring output interval
    bool save_spike_data = false;       // Save detailed spike data
    
    void print() const {
        std::cout << "=== Network Configuration ===" << std::endl;
        std::cout << "Topology: " << input_size << " -> " << hidden_size 
                  << " -> " << output_size << std::endl;
        std::cout << "Columns: " << num_columns << " x " << neurons_per_column 
                  << " neurons" << std::endl;
        std::cout << "Simulation: dt=" << dt << "ms, threshold=" 
                  << spike_threshold << "mV" << std::endl;
        std::cout << "STDP: A+=" << A_plus << ", A-=" << A_minus 
                  << ", tau+=" << tau_plus << ", tau-=" << tau_minus << std::endl;
        std::cout << "Reward: learning_rate=" << reward_learning_rate 
                  << ", decay=" << reward_decay << std::endl;
        std::cout << "=============================" << std::endl;
    }
    
    // Validation
    bool validate() const {
        if (input_size <= 0 || hidden_size <= 0 || output_size <= 0) {
            std::cerr << "ERROR: Invalid network sizes" << std::endl;
            return false;
        }
        if (num_columns <= 0 || neurons_per_column <= 0) {
            std::cerr << "ERROR: Invalid column configuration" << std::endl;
            return false;
        }
        if (dt <= 0 || dt > 1.0f) {
            std::cerr << "ERROR: Invalid time step" << std::endl;
            return false;
        }
        if (exc_ratio < 0 || exc_ratio > 1.0f) {
            std::cerr << "ERROR: Invalid excitatory ratio" << std::endl;
            return false;
        }
        return true;
    }
};

// Configuration presets for different use cases
namespace NetworkPresets {
    inline NetworkConfig trading_optimized() {
        NetworkConfig config;
        config.hidden_size = 1024;
        config.num_columns = 16;
        config.neurons_per_column = 64;
        config.A_plus = 0.008f;
        config.reward_learning_rate = 0.15f;
        config.simulation_time = 50.0f; // Faster decisions
        return config;
    }
    
    inline NetworkConfig research_detailed() {
        NetworkConfig config;
        config.enable_monitoring = true;
        config.save_spike_data = true;
        config.simulation_time = 200.0f; // Longer simulation
        config.monitoring_interval = 100;
        return config;
    }
    
    inline NetworkConfig fast_testing() {
        NetworkConfig config;
        config.hidden_size = 256;
        config.num_columns = 4;
        config.neurons_per_column = 64;
        config.simulation_time = 25.0f;
        config.enable_monitoring = false;
        return config;
    }
}

#endif // NETWORK_CONFIG_H