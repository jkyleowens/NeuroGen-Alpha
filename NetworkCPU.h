#pragma once
#ifndef NETWORK_CPU_H
#define NETWORK_CPU_H

#include <vector>
#include <memory>
#include <random>
#include "include/NeuroGen/NetworkConfig.h"

// CPU-only neural network implementation
class NetworkCPU {
private:
    struct CPUNeuronState {
        float voltage = -65.0f;
        bool spiked = false;
        float last_spike_time = -1.0f;
        float m = 0.05f, h = 0.60f, n = 0.32f;
    };
    
    struct CPUSynapse {
        int pre_neuron_idx;
        int post_neuron_idx;
        float weight;
        float delay;
        float last_pre_spike_time;
        float activity_metric;
    };
    
    std::vector<CPUNeuronState> neurons_;
    std::vector<CPUSynapse> synapses_;
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    NetworkConfig config_;
    float current_time_;
    std::mt19937 rng_;
    
public:
    NetworkCPU(const NetworkConfig& config = NetworkConfig{});
    ~NetworkCPU() = default;
    
    // Core interface matching CUDA version
    void initialize();
    std::vector<float> forward(const std::vector<float>& input, float reward_signal);
    void updateWeights(float reward_signal);
    void cleanup();
    
    // Configuration
    void setConfig(const NetworkConfig& config);
    NetworkConfig getConfig() const;
    void printStats() const;
    
    // Save/load state
    void saveState(const std::string& filename) const;
    void loadState(const std::string& filename);
    void reset();
    
private:
    void updateNeuron(CPUNeuronState& neuron, float dt, float external_current);
    void propagateSpikes(float dt);
    void applySTDP(float dt);
    std::vector<float> extractOutput();
    void createTopology();
};

#endif // NETWORK_CPU_H
