#pragma once
#include <iostream>
#include <string>
#include <vector> // Included for potential future use, e.g. string lists for layer names etc.

struct NetworkConfig {
    // Network Architecture
    int input_size = 60;
    int hidden_size = 256; // Size of a single hidden layer, or a default if multiple
    int output_size = 3;
    // std::vector<int> hidden_layer_sizes = {256, 128}; // Example for multiple hidden layers

    // Simulation Parameters
    float dt = 0.025f;              // Simulation time step (ms)
    float simulation_time = 10.0f;  // Total simulation duration (s)

    // Connectivity Parameters
    float input_hidden_prob = 0.3f;
    float hidden_hidden_prob = 0.1f; // For recurrent connections or between multiple hidden layers
    float hidden_output_prob = 1.0f;

    // Neuron Parameters (General)
    float spike_threshold = -40.0f; // mV
    // Hodgkin-Huxley specific parameters might go into a separate struct or be prefixed
    // float gNa = 120.0f; // mS/cm^2
    // float gK = 36.0f;   // mS/cm^2
    // float gL = 0.3f;    // mS/cm^2
    // float ENa = 50.0f;  // mV
    // float EK = -77.0f;  // mV
    // float EL = -54.387f;// mV

    // Synaptic Parameters
    float weight_init_std = 0.1f;   // Standard deviation for initial weight distribution
    float exc_ratio = 0.8f;         // Ratio of excitatory to inhibitory neurons
    float delay_min = 0.5f;         // Minimum synaptic delay (ms)
    float delay_max = 2.0f;         // Maximum synaptic delay (ms)
    float min_weight = -2.0f;       // Minimum allowable synaptic weight
    float max_weight = 2.0f;        // Maximum allowable synaptic weight

    // STDP Parameters
    float A_plus = 0.01f;           // LTP learning rate
    float A_minus = 0.01f;          // LTD learning rate (often A_plus * 1.05 or similar)
    float tau_plus = 20.0f;         // LTP time constant (ms)
    float tau_minus = 20.0f;        // LTD time constant (ms)

    // Input Current Parameters
    float input_current_scale = 10.0f; // Scaling factor for input currents

    // Reinforcement Learning / Reward Parameters
    float reward_learning_rate = 0.1f;

    // Homeostatic Plasticity
    float homeostatic_strength = 0.01f; // Strength of homeostatic regulation

    // Monitoring and Logging
    bool enable_monitoring = true;
    int monitoring_interval = 100;  // Timesteps or ms, define clearly
    std::string output_directory = "results/";

    // Validation method
    bool validate() const {
        if (input_size <= 0 || hidden_size <= 0 || output_size <= 0) return false;
        if (dt <= 0 || simulation_time <= 0) return false;
        if (exc_ratio < 0.0f || exc_ratio > 1.0f) return false;
        // Add more checks as needed
        return true;
    }

    // Print method for debugging
    void print() const {
        std::cout << "--- Network Configuration ---" << std::endl;
        std::cout << "  Architecture: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
        std::cout << "  Simulation: dt=" << dt << "ms, time=" << simulation_time << "s" << std::endl;
        std::cout << "  Connectivity: P(I-H)=" << input_hidden_prob << ", P(H-H)=" << hidden_hidden_prob << ", P(H-O)=" << hidden_output_prob << std::endl;
        std::cout << "  Spike Threshold: " << spike_threshold << "mV" << std::endl;
        std::cout << "  STDP: A+ = " << A_plus << ", A- = " << A_minus << ", tau+ = " << tau_plus << "ms, tau- = " << tau_minus << "ms" << std::endl;
        if (enable_monitoring) {
            std::cout << "  Monitoring: Enabled, Interval=" << monitoring_interval << ", Output Dir=" << output_directory << std::endl;
        } else {
            std::cout << "  Monitoring: Disabled" << std::endl;
        }
        std::cout << "--- End Configuration ---" << std::endl;
    }

    // Method to finalize or derive any dependent parameters
    void finalizeConfig() {
        // Example: if simulation_time is in seconds, convert to milliseconds or timesteps
        // num_timesteps = static_cast<int>(simulation_time * 1000.0f / dt);
    }
};

namespace NetworkPresets {
    inline NetworkConfig get_default_config() {
        return NetworkConfig(); // Returns the config with default initialized values
    }

    inline NetworkConfig trading_optimized() {
        NetworkConfig config;
        // Modify specific parameters for a trading-optimized setup
        config.input_size = 75; // e.g., more features for trading
        config.hidden_size = 512;
        config.output_size = 3; // Buy, Sell, Hold
        config.simulation_time = 60.0f; // Longer simulation for trading scenarios
        config.A_plus = 0.005f;
        config.A_minus = 0.0055f;
        config.reward_learning_rate = 0.05f;
        config.enable_monitoring = true;
        config.monitoring_interval = 500;
        return config;
    }

    inline NetworkConfig fast_debug_config() {
        NetworkConfig config;
        config.simulation_time = 1.0f; // Short simulation
        config.hidden_size = 32;
        config.input_hidden_prob = 0.5f;
        config.enable_monitoring = false;
        return config;
    }
}