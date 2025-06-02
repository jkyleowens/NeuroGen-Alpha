#pragma once
#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <iostream>
#include <string>
#include <stdexcept>

// Network configuration structure
struct NetworkConfig {
    // Network topology
    int input_size = 60;
    int hidden_size = 256;
    int output_size = 3;
    
    // Simulation parameters
    float dt = 0.01f;                    // Time step (ms)
    float simulation_time = 50.0f;       // Total simulation time per forward pass (ms)
    float spike_threshold = -40.0f;      // Spike threshold (mV)
    float input_current_scale = 20.0f;   // Input current scaling factor
    
    // Weight initialization
    float weight_init_std = 0.1f;        // Standard deviation for weight initialization
    float delay_min = 0.5f;              // Minimum synaptic delay (ms)
    float delay_max = 3.0f;              // Maximum synaptic delay (ms)
    
    // Connection probabilities
    float input_hidden_prob = 0.8f;      // Probability of input->hidden connections
    float hidden_hidden_prob = 0.1f;     // Probability of hidden->hidden connections
    float exc_ratio = 0.8f;              // Ratio of excitatory to inhibitory connections
    
    // STDP parameters
    float A_plus = 0.01f;                // Potentiation amplitude
    float A_minus = 0.012f;              // Depression amplitude
    float tau_plus = 20.0f;              // Potentiation time constant (ms)
    float tau_minus = 20.0f;             // Depression time constant (ms)
    float min_weight = -2.0f;            // Minimum synaptic weight
    float max_weight = 2.0f;             // Maximum synaptic weight
    
    // Learning parameters
    float reward_learning_rate = 0.1f;   // Reward signal learning rate
    float homeostatic_strength = 0.001f; // Homeostatic scaling strength
    
    // Monitoring
    bool enable_monitoring = true;       // Enable performance monitoring
    int monitoring_interval = 100;      // Monitoring update interval
    
    // Validation function
    bool validate() const {
        if (input_size <= 0 || hidden_size <= 0 || output_size <= 0) {
            std::cerr << "[ERROR] Network sizes must be positive" << std::endl;
            return false;
        }
        
        if (dt <= 0.0f || dt > 1.0f) {
            std::cerr << "[ERROR] Time step must be in range (0, 1] ms" << std::endl;
            return false;
        }
        
        if (simulation_time <= 0.0f || simulation_time > 1000.0f) {
            std::cerr << "[ERROR] Simulation time must be in range (0, 1000] ms" << std::endl;
            return false;
        }
        
        if (spike_threshold >= 0.0f || spike_threshold < -100.0f) {
            std::cerr << "[ERROR] Spike threshold must be in range [-100, 0) mV" << std::endl;
            return false;
        }
        
        if (weight_init_std <= 0.0f || weight_init_std > 1.0f) {
            std::cerr << "[ERROR] Weight initialization std must be in range (0, 1]" << std::endl;
            return false;
        }
        
        if (delay_min <= 0.0f || delay_max <= delay_min || delay_max > 100.0f) {
            std::cerr << "[ERROR] Invalid delay range" << std::endl;
            return false;
        }
        
        if (input_hidden_prob < 0.0f || input_hidden_prob > 1.0f ||
            hidden_hidden_prob < 0.0f || hidden_hidden_prob > 1.0f ||
            exc_ratio < 0.0f || exc_ratio > 1.0f) {
            std::cerr << "[ERROR] Probabilities must be in range [0, 1]" << std::endl;
            return false;
        }
        
        if (A_plus <= 0.0f || A_minus <= 0.0f ||
            tau_plus <= 0.0f || tau_minus <= 0.0f) {
            std::cerr << "[ERROR] STDP parameters must be positive" << std::endl;
            return false;
        }
        
        if (min_weight >= max_weight) {
            std::cerr << "[ERROR] min_weight must be less than max_weight" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Print configuration
    void print() const {
        std::cout << "[CONFIG] Network topology: " << input_size << " -> " 
                  << hidden_size << " -> " << output_size << std::endl;
        std::cout << "[CONFIG] Simulation: dt=" << dt << "ms, time=" 
                  << simulation_time << "ms" << std::endl;
        std::cout << "[CONFIG] Spike threshold: " << spike_threshold << "mV" << std::endl;
        std::cout << "[CONFIG] STDP: A+=" << A_plus << ", A-=" << A_minus 
                  << ", tau+=" << tau_plus << "ms, tau-=" << tau_minus << "ms" << std::endl;
        std::cout << "[CONFIG] Weights: [" << min_weight << ", " << max_weight << "]" << std::endl;
        std::cout << "[CONFIG] Connection probs: I->H=" << input_hidden_prob 
                  << ", H->H=" << hidden_hidden_prob << std::endl;
        std::cout << "[CONFIG] E/I ratio: " << exc_ratio << std::endl;
    }
    
    // Save configuration to file
    void save(const std::string& filename) const {
        // Implementation would save config parameters to file
        std::cout << "[CONFIG] Saving configuration to " << filename << std::endl;
    }
    
    // Load configuration from file
    void load(const std::string& filename) {
        // Implementation would load config parameters from file
        std::cout << "[CONFIG] Loading configuration from " << filename << std::endl;
    }
};

// Predefined network configurations
namespace NetworkPresets {
    
    // Small network for testing
    inline NetworkConfig small_test() {
        NetworkConfig config;
        config.input_size = 10;
        config.hidden_size = 32;
        config.output_size = 3;
        config.simulation_time = 20.0f;
        config.input_current_scale = 10.0f;
        return config;
    }
    
    // Medium network for development
    inline NetworkConfig medium_dev() {
        NetworkConfig config;
        config.input_size = 30;
        config.hidden_size = 128;
        config.output_size = 3;
        config.simulation_time = 35.0f;
        config.input_current_scale = 15.0f;
        return config;
    }
    
    // Large network for production
    inline NetworkConfig large_production() {
        NetworkConfig config;
        config.input_size = 100;
        config.hidden_size = 512;
        config.output_size = 5;
        config.simulation_time = 75.0f;
        config.input_current_scale = 25.0f;
        config.hidden_hidden_prob = 0.05f; // Sparser connections for large network
        return config;
    }
    
    // Trading-optimized network (default for financial applications)
    inline NetworkConfig trading_optimized() {
        NetworkConfig config;
        config.input_size = 60;              // Technical indicators
        config.hidden_size = 256;            // Good balance of capacity/speed
        config.output_size = 3;              // Buy/Hold/Sell
        config.simulation_time = 25.0f;      // Fast decisions
        config.input_current_scale = 15.0f;  // Moderate input strength
        config.spike_threshold = -45.0f;     // Slightly higher threshold for stability
        config.reward_learning_rate = 0.15f; // Faster adaptation to rewards
        config.A_plus = 0.015f;              // Stronger potentiation for profitable strategies
        config.A_minus = 0.010f;             // Gentler depression to preserve good strategies
        config.homeostatic_strength = 0.002f; // Stronger homeostasis for stability
        return config;
    }
    
    // Research network with enhanced monitoring
    inline NetworkConfig research_enhanced() {
        NetworkConfig config = trading_optimized();
        config.enable_monitoring = true;
        config.monitoring_interval = 50;     // More frequent monitoring
        config.simulation_time = 100.0f;     // Longer simulation for detailed analysis
        return config;
    }
    
    // Fast inference network (minimal simulation time)
    inline NetworkConfig fast_inference() {
        NetworkConfig config;
        config.input_size = 40;
        config.hidden_size = 128;
        config.output_size = 3;
        config.simulation_time = 15.0f;      // Minimal simulation time
        config.dt = 0.025f;                  // Larger time step
        config.input_current_scale = 25.0f;  // Stronger input to compensate
        config.enable_monitoring = false;    // Disable monitoring for speed
        return config;
    }
}

// Configuration builder helper class
class NetworkConfigBuilder {
private:
    NetworkConfig config_;
    
public:
    NetworkConfigBuilder() = default;
    NetworkConfigBuilder(const NetworkConfig& base) : config_(base) {}
    
    NetworkConfigBuilder& inputSize(int size) { config_.input_size = size; return *this; }
    NetworkConfigBuilder& hiddenSize(int size) { config_.hidden_size = size; return *this; }
    NetworkConfigBuilder& outputSize(int size) { config_.output_size = size; return *this; }
    NetworkConfigBuilder& timeStep(float dt) { config_.dt = dt; return *this; }
    NetworkConfigBuilder& simulationTime(float time) { config_.simulation_time = time; return *this; }
    NetworkConfigBuilder& spikeThreshold(float threshold) { config_.spike_threshold = threshold; return *this; }
    NetworkConfigBuilder& inputScale(float scale) { config_.input_current_scale = scale; return *this; }
    NetworkConfigBuilder& weightStd(float std) { config_.weight_init_std = std; return *this; }
    NetworkConfigBuilder& delays(float min, float max) { 
        config_.delay_min = min; 
        config_.delay_max = max; 
        return *this; 
    }
    NetworkConfigBuilder& connectionProbs(float input_hidden, float hidden_hidden) {
        config_.input_hidden_prob = input_hidden;
        config_.hidden_hidden_prob = hidden_hidden;
        return *this;
    }
    NetworkConfigBuilder& excRatio(float ratio) { config_.exc_ratio = ratio; return *this; }
    NetworkConfigBuilder& stdpParams(float A_plus, float A_minus, float tau_plus, float tau_minus) {
        config_.A_plus = A_plus;
        config_.A_minus = A_minus;
        config_.tau_plus = tau_plus;
        config_.tau_minus = tau_minus;
        return *this;
    }
    NetworkConfigBuilder& weightBounds(float min, float max) {
        config_.min_weight = min;
        config_.max_weight = max;
        return *this;
    }
    NetworkConfigBuilder& learningRate(float rate) { config_.reward_learning_rate = rate; return *this; }
    NetworkConfigBuilder& homeostaticStrength(float strength) { config_.homeostatic_strength = strength; return *this; }
    NetworkConfigBuilder& monitoring(bool enable, int interval = 100) {
        config_.enable_monitoring = enable;
        config_.monitoring_interval = interval;
        return *this;
    }
    
    NetworkConfig build() const {
        NetworkConfig result = config_;
        if (!result.validate()) {
            throw std::invalid_argument("Invalid network configuration");
        }
        return result;
    }
};

#endif // NETWORK_CONFIG_H