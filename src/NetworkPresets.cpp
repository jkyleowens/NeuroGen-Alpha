#include "NetworkPresets.h"

namespace NetworkPresets {

    NetworkConfig trading_optimized() {
        NetworkConfig config;
        
        // Network topology optimized for trading
        config.input_size = 64;        // Market indicators, price data, volume, etc.
        config.hidden_size = 256;      // Sufficient capacity for pattern recognition
        config.output_size = 10;       // Trading decisions (buy/sell/hold for multiple assets)
        
        // Trading-specific timing
        config.dt = 0.001;             // Fast update rate for market responsiveness
        config.simulation_time = 100.0f; // Extended simulation for strategy development
        
        // Learning parameters optimized for financial data
        config.reward_learning_rate = 0.005f;  // Conservative learning for stability
        config.A_plus = 0.008f;                // Moderate potentiation
        config.A_minus = 0.010f;               // Slightly stronger depression
        config.tau_plus = 15.0f;               // Fast potentiation window
        config.tau_minus = 15.0f;              // Fast depression window
        
        // Connection probabilities for trading networks
        config.input_hidden_prob = 0.85f;     // Strong input connectivity
        config.hidden_hidden_prob = 0.15f;    // Moderate recurrence
        config.hidden_output_prob = 0.95f;    // Strong output connectivity
        
        // Weight parameters
        config.min_weight = 0.001f;
        config.max_weight = 3.0f;              // Higher max weight for strong signals
        config.weight_init_std = 0.6f;         // Moderate initial variance
        
        // Excitatory/inhibitory balance
        config.exc_ratio = 0.85f;              // Higher excitatory ratio for trading
        
        // Homeostatic regulation
        config.homeostatic_strength = 0.0005f; // Gentle homeostasis
        
        // Monitoring
        config.enable_monitoring = true;
        config.monitoring_interval = 50;       // Frequent monitoring for trading
        
        // Spike dynamics
        config.spike_threshold = 25.0;         // Lower threshold for responsiveness
        
        // Finalize configuration to compute derived values
        config.finalizeConfig();
        
        return config;
    }

    NetworkConfig high_frequency_trading() {
        NetworkConfig config;
        
        // Compact network for speed
        config.input_size = 32;        // Essential market data only
        config.hidden_size = 128;      // Smaller hidden layer for speed
        config.output_size = 5;        // Simple decisions (strong buy/buy/hold/sell/strong sell)
        
        // Ultra-fast timing for HFT
        config.dt = 0.0005;            // Very fast update rate
        config.simulation_time = 50.0f; // Shorter simulation windows
        
        // Aggressive learning for quick adaptation
        config.reward_learning_rate = 0.01f;   // Fast learning
        config.A_plus = 0.015f;                // Strong potentiation
        config.A_minus = 0.018f;               // Strong depression
        config.tau_plus = 10.0f;               // Very fast plasticity windows
        config.tau_minus = 10.0f;
        
        // Dense connectivity for speed
        config.input_hidden_prob = 0.9f;      // Maximum input connectivity
        config.hidden_hidden_prob = 0.05f;    // Minimal recurrence for speed
        config.hidden_output_prob = 0.98f;    // Near-complete output connectivity
        
        // Weight parameters for HFT
        config.min_weight = 0.002f;
        config.max_weight = 2.5f;
        config.weight_init_std = 0.4f;         // Lower variance for stability
        
        // High excitatory ratio for quick responses
        config.exc_ratio = 0.9f;
        
        // Minimal homeostasis for speed
        config.homeostatic_strength = 0.0001f;
        
        // Minimal monitoring to reduce overhead
        config.enable_monitoring = true;
        config.monitoring_interval = 100;
        
        // Low spike threshold for high sensitivity
        config.spike_threshold = 20.0;

        config.finalizeConfig();
        return config;
    }

    NetworkConfig research_detailed() {
        NetworkConfig config;
        
        // Large network for detailed research
        config.input_size = 128;       // Comprehensive input space
        config.hidden_size = 512;      // Large hidden layer for complex patterns
        config.output_size = 20;       // Detailed output classifications
        
        // Research-appropriate timing
        config.dt = 0.01;              // Standard research time step
        config.simulation_time = 200.0f; // Extended simulation for analysis
        
        // Conservative learning for detailed study
        config.reward_learning_rate = 0.002f;  // Slow, careful learning
        config.A_plus = 0.005f;                // Moderate plasticity
        config.A_minus = 0.006f;
        config.tau_plus = 25.0f;               // Longer plasticity windows
        config.tau_minus = 25.0f;
        
        // Research-typical connectivity
        config.input_hidden_prob = 0.7f;      // Biological-like sparsity
        config.hidden_hidden_prob = 0.2f;     // Rich recurrent structure
        config.hidden_output_prob = 0.8f;     // Strong but not complete connectivity
        
        // Standard weight parameters
        config.min_weight = 0.001f;
        config.max_weight = 2.0f;
        config.weight_init_std = 0.5f;
        
        // Biological excitatory ratio
        config.exc_ratio = 0.8f;
        
        // Active homeostasis for long simulations
        config.homeostatic_strength = 0.001f;
        
        // Detailed monitoring for research
        config.enable_monitoring = true;
        config.monitoring_interval = 25;
        
        // Standard spike threshold
        config.spike_threshold = 30.0;

        config.finalizeConfig();
        return config;
    }

    NetworkConfig minimal_test() {
        NetworkConfig config;
        
        // Tiny network for quick testing
        config.input_size = 8;         // Minimal input
        config.hidden_size = 16;       // Small hidden layer
        config.output_size = 3;        // Simple output
        
        // Fast testing timing
        config.dt = 0.01;
        config.simulation_time = 20.0f; // Short test runs
        
        // Standard learning parameters
        config.reward_learning_rate = 0.01f;
        config.A_plus = 0.01f;
        config.A_minus = 0.012f;
        config.tau_plus = 20.0f;
        config.tau_minus = 20.0f;
        
        // High connectivity for small network
        config.input_hidden_prob = 0.95f;
        config.hidden_hidden_prob = 0.3f;
        config.hidden_output_prob = 0.95f;
        
        // Standard weights
        config.min_weight = 0.001f;
        config.max_weight = 2.0f;
        config.weight_init_std = 0.5f;
        
        // Standard excitatory ratio
        config.exc_ratio = 0.8f;
        
        // Minimal homeostasis
        config.homeostatic_strength = 0.001f;
        
        // Basic monitoring
        config.enable_monitoring = true;
        config.monitoring_interval = 100;
        
        // Standard threshold
        config.spike_threshold = 30.0;

        config.finalizeConfig();
        return config;
    }

    NetworkConfig balanced_default() {
        NetworkConfig config;
        
        // Balanced network size
        config.input_size = 32;        // Moderate input size
        config.hidden_size = 128;      // Balanced hidden layer
        config.output_size = 8;        // Reasonable output size
        
        // Standard timing
        config.dt = 0.01;
        config.simulation_time = 100.0f;
        
        // Balanced learning parameters
        config.reward_learning_rate = 0.01f;
        config.A_plus = 0.01f;
        config.A_minus = 0.012f;
        config.tau_plus = 20.0f;
        config.tau_minus = 20.0f;
        
        // Balanced connectivity
        config.input_hidden_prob = 0.8f;
        config.hidden_hidden_prob = 0.1f;
        config.hidden_output_prob = 0.9f;
        
        // Standard weights
        config.min_weight = 0.001f;
        config.max_weight = 2.0f;
        config.weight_init_std = 0.5f;
        
        // Biological excitatory ratio
        config.exc_ratio = 0.8f;
        
        // Standard homeostasis
        config.homeostatic_strength = 0.001f;
        
        // Regular monitoring
        config.enable_monitoring = true;
        config.monitoring_interval = 100;
        
        // Standard threshold
        config.spike_threshold = 30.0;

        config.finalizeConfig();
        return config;
    }

    NetworkConfig getSmallNetworkConfig() {
        NetworkConfig config;
        
        // Small network topology for testing
        config.input_size = 16;        // Small input
        config.hidden_size = 32;       // Small hidden layer
        config.output_size = 4;        // Limited output
        
        // Standard timing
        config.dt = 0.001;             // Standard time step
        config.simulation_time = 50.0f; // Short simulation for testing
        
        // Moderate learning parameters
        config.reward_learning_rate = 0.01f;
        config.A_plus = 0.01f;
        config.A_minus = 0.012f;
        config.tau_plus = 20.0f;
        config.tau_minus = 20.0f;
        
        // Standard connectivity
        config.input_hidden_prob = 0.8f;
        config.hidden_hidden_prob = 0.1f;
        config.hidden_output_prob = 0.9f;
        
        // Standard weights
        config.min_weight = 0.001f;
        config.max_weight = 2.0f;
        config.weight_init_std = 0.5f;
        
        // Standard excitatory ratio
        config.exc_ratio = 0.8f;
        
        // Standard homeostasis
        config.homeostatic_strength = 0.001f;
        
        // Monitoring
        config.enable_monitoring = true;
        config.monitoring_interval = 100;
        
        // Standard threshold
        config.spike_threshold = 30.0;

        config.finalizeConfig();
        return config;
    }


} // namespace NetworkPresets
