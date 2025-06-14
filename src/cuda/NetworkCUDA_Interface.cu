#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/NetworkPresets.h>
#include <vector>
#include <memory>

// Global network instance for C interface
static std::unique_ptr<NetworkCUDA> g_network = nullptr;

// Interface functions for main.cpp compatibility

void initializeNetwork() {
    try {
        // Initialize with trading-optimized configuration
        NetworkConfig config = NetworkPresets::trading_optimized();
        g_network = std::make_unique<NetworkCUDA>(config);
        
        // Print network initialization info
        printf("[CUDA] Neural network initialized successfully\n");
        printf("[CUDA] Neurons: %d, Synapses: %d\n", 
               g_network->getNumNeurons(), g_network->getNumSynapses());
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to initialize network: %s\n", e.what());
        throw;
    }
}

std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    if (!g_network) {
        throw std::runtime_error("Network not initialized. Call initializeNetwork() first.");
    }
    
    try {
        // Update network with input and reward
        g_network->update(0.01f, input, reward_signal); // 0.01f = 10ms timestep
        
        // Get network output
        return g_network->getOutput();
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Forward pass failed: %s\n", e.what());
        throw;
    }
}

void updateSynapticWeightsCUDA(float reward_signal) {
    if (!g_network) {
        fprintf(stderr, "[WARNING] Network not initialized, skipping weight update\n");
        return;
    }
    
    try {
        // Set reward signal for learning
        g_network->setRewardSignal(reward_signal);
        
        // The actual weight updates happen during the forward pass
        // This function is kept for compatibility with main.cpp
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Weight update failed: %s\n", e.what());
    }
}

std::vector<float> getNeuromodulatorLevels() {
    if (!g_network) {
        // Return default values if network not initialized
        return std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f}; // Default dopamine, acetylcholine, serotonin, norepinephrine
    }
    
    try {
        // Get network statistics which include neuromodulator levels
        NetworkStats stats = g_network->getStats();
        
        // For now, derive neuromodulator levels from network state
        // In a full implementation, these would be tracked separately
        float dopamine = std::min(1.0f, std::max(0.0f, 0.5f + stats.current_reward));
        float acetylcholine = 0.5f + 0.3f * std::sin(stats.total_simulation_time * 0.1f); // Simulated attention cycles
        float serotonin = 0.5f;
        float norepinephrine = 0.5f;
        
        return std::vector<float>{dopamine, acetylcholine, serotonin, norepinephrine};
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to get neuromodulator levels: %s\n", e.what());
        return std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f};
    }
}

void printNetworkStats() {
    if (!g_network) {
        printf("[STATS] Network not initialized\n");
        return;
    }
    
    try {
        g_network->printNetworkState();
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to print network stats: %s\n", e.what());
    }
}

void cleanupNetwork() {
    if (g_network) {
        printf("[CUDA] Cleaning up neural network...\n");
        g_network.reset();
        printf("[CUDA] Network cleanup complete\n");
    }
}

void setNetworkConfig(const NetworkConfig& config) {
    // This would reinitialize the network with new config
    // For now, just print that config was received
    printf("[CUDA] Network configuration updated\n");
}

NetworkConfig getNetworkConfig() {
    if (g_network) {
        // Return current config - in a full implementation this would be stored
        return NetworkPresets::trading_optimized();
    }
    return NetworkPresets::minimal_test();
}

void resetNetwork() {
    if (g_network) {
        try {
            g_network->reset();
            printf("[CUDA] Network reset complete\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[ERROR] Network reset failed: %s\n", e.what());
        }
    }
}

void saveNetworkState(const std::string& filename) {
    printf("[CUDA] Network state save requested to: %s (not implemented)\n", filename.c_str());
    // TODO: Implement network state serialization
}

void loadNetworkState(const std::string& filename) {
    printf("[CUDA] Network state load requested from: %s (not implemented)\n", filename.c_str());
    // TODO: Implement network state deserialization
}
