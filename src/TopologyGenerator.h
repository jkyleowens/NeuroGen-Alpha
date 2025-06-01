#pragma once
#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include <vector>
#include <random>
#include "GPUNeuralStructures.h"
#include "CorticalColumn.h"

// Network configuration structure
struct NetworkConfig {
    // Basic network dimensions
    int numColumns = 4;
    int neuronsPerColumn = 256;
    float dt = 0.025f;  // Integration time step (ms)
    
    // Connectivity parameters
    float excRatio = 0.8f;      // Fraction of excitatory neurons
    int localFanIn = 30;        // Incoming connections per neuron
    int localFanOut = 30;       // Outgoing connections per neuron
    
    // Synaptic weight ranges
    float wExcMin = 0.05f;      // Minimum excitatory weight
    float wExcMax = 0.15f;      // Maximum excitatory weight
    float wInhMin = 0.20f;      // Minimum inhibitory weight magnitude
    float wInhMax = 0.40f;      // Maximum inhibitory weight magnitude
    
    // Synaptic delay ranges
    float dMin = 0.5f;          // Minimum delay (ms)
    float dMax = 2.0f;          // Maximum delay (ms)
    
    // Computed values (set by finalizeConfig)
    size_t totalSynapses = 0;   // Total number of synapses
    
    // Configuration validation and finalization
    bool validate() const {
        return (numColumns > 0 && neuronsPerColumn > 0 && 
                excRatio >= 0.0f && excRatio <= 1.0f &&
                localFanIn > 0 && localFanOut > 0 &&
                dt > 0.0f && dt <= 1.0f);
    }
    
    void finalizeConfig() {
        if (!validate()) {
            throw std::runtime_error("Invalid network configuration");
        }
        totalSynapses = static_cast<size_t>(numColumns) * neuronsPerColumn * localFanOut;
    }
};

// Topology generator class for creating network connectivity
class TopologyGenerator {
private:
    const NetworkConfig& cfg_;
    std::mt19937 rng_;
    
public:
    explicit TopologyGenerator(const NetworkConfig& cfg);
    
    // Build local recurrent connections within columns
    void buildLocalLoops(std::vector<GPUSynapse>& synapses,
                        const std::vector<GPUCorticalColumn>& columns);
    
    // Build inter-column connections (for future use)
    void buildInterColumnConnections(std::vector<GPUSynapse>& synapses,
                                   const std::vector<GPUCorticalColumn>& columns,
                                   float connection_probability = 0.1f);
    
    // Build feedforward connections from input layer
    void buildInputConnections(std::vector<GPUSynapse>& synapses,
                             int input_start, int input_end,
                             int target_start, int target_end,
                             float connection_probability = 0.8f);
    
    // Utility functions
    void shuffleConnections(std::vector<GPUSynapse>& synapses);
    void validateTopology(const std::vector<GPUSynapse>& synapses,
                         const std::vector<GPUCorticalColumn>& columns) const;
    
    // Statistics
    void printTopologyStats(const std::vector<GPUSynapse>& synapses,
                           const std::vector<GPUCorticalColumn>& columns) const;
    
private:
    // Helper functions
    bool isExcitatoryNeuron(int neuron_idx, const GPUCorticalColumn& column) const;
    float generateSynapticWeight(bool is_excitatory) const;
    float generateSynapticDelay() const;
    int selectRandomTarget(const GPUCorticalColumn& column) const;
};

// Utility function to finalize network configuration
inline void finalizeConfig(NetworkConfig& cfg) {
    cfg.finalizeConfig();
}

// Default configurations for different use cases
namespace ConfigPresets {
    NetworkConfig smallNetwork();
    NetworkConfig mediumNetwork();
    NetworkConfig largeNetwork();
    NetworkConfig testNetwork();
}

#endif // TOPOLOGY_GENERATOR_H