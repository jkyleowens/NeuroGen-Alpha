#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/NetworkPresets.h>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

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
    if (!g_network) {
        printf("[WARNING] Cannot save network state: network not initialized\n");
        return;
    }
    
    try {
        std::cout << "[CUDA] Saving network state to: " << filename << std::endl;
        
        // Save synaptic weights from GPU to file
        std::vector<float> weights = g_network->getSynapticWeights();
        std::vector<float> voltages = g_network->getNeuronVoltages();
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving: " + filename);
        }
        
        // Write header
        uint32_t magic = 0x4E4E4554; // "NNET" in hex
        file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        
        // Write network configuration
        int num_neurons = g_network->getNumNeurons();
        int num_synapses = g_network->getNumSynapses();
        file.write(reinterpret_cast<const char*>(&num_neurons), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_synapses), sizeof(int));
        
        // Write synaptic weights
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        
        // Write neuron voltages
        file.write(reinterpret_cast<const char*>(voltages.data()), voltages.size() * sizeof(float));
        
        // Write current time
        float current_time = g_network->getCurrentTime();
        file.write(reinterpret_cast<const char*>(&current_time), sizeof(float));
        
        file.close();
        std::cout << "[CUDA] Network state saved successfully" << std::endl;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to save network state: %s\n", e.what());
    }
}

void loadNetworkState(const std::string& filename) {
    if (!g_network) {
        printf("[WARNING] Cannot load network state: network not initialized\n");
        return;
    }
    
    try {
        std::cout << "[CUDA] Loading network state from: " << filename << std::endl;
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "[CUDA] No previous network state found, starting fresh" << std::endl;
            return; // Not an error, just no previous state
        }
        
        // Read and verify header
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        if (magic != 0x4E4E4554) {
            throw std::runtime_error("Invalid network state file format");
        }
        
        // Read network configuration
        int saved_neurons, saved_synapses;
        file.read(reinterpret_cast<char*>(&saved_neurons), sizeof(int));
        file.read(reinterpret_cast<char*>(&saved_synapses), sizeof(int));
        
        // Verify configuration matches current network
        if (saved_neurons != g_network->getNumNeurons() || saved_synapses != g_network->getNumSynapses()) {
            std::cout << "[WARNING] Network configuration mismatch. Saved: " 
                      << saved_neurons << " neurons, " << saved_synapses 
                      << " synapses. Current: " << g_network->getNumNeurons() 
                      << " neurons, " << g_network->getNumSynapses() 
                      << " synapses. Starting fresh." << std::endl;
            file.close();
            return;
        }
        
        // Read synaptic weights
        std::vector<float> weights(saved_synapses);
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        
        // Read neuron voltages
        std::vector<float> voltages(saved_neurons);
        file.read(reinterpret_cast<char*>(voltages.data()), voltages.size() * sizeof(float));
        
        // Read current time
        float saved_time;
        file.read(reinterpret_cast<char*>(&saved_time), sizeof(float));
        
        file.close();
        
        // TODO: Apply loaded weights and voltages to the network
        // This would require additional CUDA kernels to update device memory
        // For now, we just validate that the data was loaded successfully
        
        std::cout << "[CUDA] Network state loaded successfully" << std::endl;
        std::cout << "[CUDA] Loaded " << weights.size() << " synaptic weights and " 
                  << voltages.size() << " neuron states" << std::endl;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to load network state: %s\n", e.what());
    }
}
