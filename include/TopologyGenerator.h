#pragma once
#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include <vector>
#include <random>
#include <iostream>
#include "NetworkConfig.h"
#include "NeuroGen/GPUNeuralStructures.h"
#include "NeuroGen/cuda/CorticalColumn.h"

// Forward declarations
struct GPUSynapse;
struct GPUCorticalColumn;

/**
 * Generates network topology and connectivity patterns for cortical neural networks
 */
class TopologyGenerator {
private:
    NetworkConfig cfg_;
    mutable std::mt19937 rng_;
    
    // Helper functions
    bool isExcitatoryNeuron(int neuron_idx, const GPUCorticalColumn& column) const;
    float generateSynapticWeight(bool is_excitatory) const;
    float generateSynapticDelay() const;
    int selectRandomTarget(const GPUCorticalColumn& column) const;
    
public:
    /**
     * Constructor
     * @param cfg Network configuration parameters
     */
    explicit TopologyGenerator(const NetworkConfig& cfg);
    
    /**
     * Build local recurrent connections within each cortical column
     * @param synapses Output vector to store generated synapses
     * @param columns Vector of cortical columns to connect
     */
    void buildLocalLoops(std::vector<GPUSynapse>& synapses,
                        const std::vector<GPUCorticalColumn>& columns);
    
    /**
     * Build sparse inter-column connections
     * @param synapses Output vector to store generated synapses
     * @param columns Vector of cortical columns to connect
     * @param connection_probability Probability of connection between columns
     */
    void buildInterColumnConnections(std::vector<GPUSynapse>& synapses,
                                   const std::vector<GPUCorticalColumn>& columns,
                                   float connection_probability = 0.1f);
    
    /**
     * Build connections from input layer to network
     * @param synapses Output vector to store generated synapses
     * @param input_start Starting index of input neurons
     * @param input_end Ending index of input neurons
     * @param target_start Starting index of target neurons
     * @param target_end Ending index of target neurons
     * @param connection_probability Probability of connection
     */
    void buildInputConnections(std::vector<GPUSynapse>& synapses,
                             int input_start, int input_end,
                             int target_start, int target_end,
                             float connection_probability = 0.8f);
    
    /**
     * Shuffle connections to improve randomization
     * @param synapses Vector of synapses to shuffle
     */
    void shuffleConnections(std::vector<GPUSynapse>& synapses);
    
    /**
     * Validate topology for consistency
     * @param synapses Vector of synapses to validate
     * @param columns Vector of cortical columns
     */
    void validateTopology(const std::vector<GPUSynapse>& synapses,
                         const std::vector<GPUCorticalColumn>& columns) const;
    
    /**
     * Print topology statistics
     * @param synapses Vector of synapses
     * @param columns Vector of cortical columns
     */
    void printTopologyStats(const std::vector<GPUSynapse>& synapses,
                           const std::vector<GPUCorticalColumn>& columns) const;
    
    /**
     * Generate distance-based connection probability
     * @param source_column Source column index
     * @param target_column Target column index
     * @param max_distance Maximum connection distance
     * @return Connection probability [0, 1]
     */
    float calculateDistanceBasedProbability(int source_column, int target_column,
                                           float max_distance = 5.0f) const;
    
    /**
     * Generate layer-specific connection patterns
     * @param synapses Output vector to store generated synapses
     * @param columns Vector of cortical columns
     * @param source_layer Source cortical layer
     * @param target_layer Target cortical layer
     * @param connection_strength Base connection strength
     */
    void buildLayerSpecificConnections(std::vector<GPUSynapse>& synapses,
                                     const std::vector<GPUCorticalColumn>& columns,
                                     int source_layer, int target_layer,
                                     float connection_strength = 1.0f);
    
    /**
     * Set random seed for reproducible topology generation
     * @param seed Random seed value
     */
    void setSeed(unsigned int seed) { rng_.seed(seed); }
    
    /**
     * Get current network configuration
     * @return Reference to network configuration
     */
    const NetworkConfig& getConfig() const { return cfg_; }
};

/**
 * Configuration presets for different network types
 */
namespace ConfigPresets {
    /**
     * Small network configuration for testing
     * @return NetworkConfig for small network
     */
    NetworkConfig smallNetwork();
    
    /**
     * Medium network configuration for development
     * @return NetworkConfig for medium network
     */
    NetworkConfig mediumNetwork();
    
    /**
     * Large network configuration for production
     * @return NetworkConfig for large network
     */
    NetworkConfig largeNetwork();
    
    /**
     * Test network configuration with fast simulation
     * @return NetworkConfig for testing
     */
    NetworkConfig testNetwork();
}

/**
 * Topology analysis utilities
 */
namespace TopologyAnalysis {
    /**
     * Calculate clustering coefficient of network
     * @param synapses Vector of synapses
     * @param num_neurons Total number of neurons
     * @return Average clustering coefficient
     */
    float calculateClusteringCoefficient(const std::vector<GPUSynapse>& synapses,
                                       int num_neurons);
    
    /**
     * Calculate average path length of network
     * @param synapses Vector of synapses
     * @param num_neurons Total number of neurons
     * @return Average path length
     */
    float calculateAveragePathLength(const std::vector<GPUSynapse>& synapses,
                                   int num_neurons);
    
    /**
     * Calculate degree distribution of network
     * @param synapses Vector of synapses
     * @param num_neurons Total number of neurons
     * @return Vector of degree counts
     */
    std::vector<int> calculateDegreeDistribution(const std::vector<GPUSynapse>& synapses,
                                                int num_neurons);
    
    /**
     * Check if network has small-world properties
     * @param synapses Vector of synapses
     * @param num_neurons Total number of neurons
     * @return True if network exhibits small-world characteristics
     */
    bool isSmallWorldNetwork(const std::vector<GPUSynapse>& synapses,
                           int num_neurons);
}

/**
 * Specialized topology generators
 */
namespace SpecializedTopologies {
    /**
     * Generate scale-free network topology
     * @param synapses Output vector for synapses
     * @param num_neurons Number of neurons
     * @param avg_degree Average degree per neuron
     * @param power_exponent Power law exponent
     */
    void generateScaleFreeTopology(std::vector<GPUSynapse>& synapses,
                                 int num_neurons, int avg_degree = 10,
                                 float power_exponent = -2.5f);
    
    /**
     * Generate small-world network topology
     * @param synapses Output vector for synapses
     * @param num_neurons Number of neurons
     * @param k Number of nearest neighbors in ring lattice
     * @param p Rewiring probability
     */
    void generateSmallWorldTopology(std::vector<GPUSynapse>& synapses,
                                  int num_neurons, int k = 6, float p = 0.1f);
    
    /**
     * Generate modular network topology
     * @param synapses Output vector for synapses
     * @param num_neurons Number of neurons
     * @param num_modules Number of modules
     * @param intra_prob Intra-module connection probability
     * @param inter_prob Inter-module connection probability
     */
    void generateModularTopology(std::vector<GPUSynapse>& synapses,
                               int num_neurons, int num_modules = 4,
                               float intra_prob = 0.3f, float inter_prob = 0.05f);
}

/**
 * Topology optimization utilities
 */
namespace TopologyOptimization {
    /**
     * Optimize topology for information flow
     * @param synapses Vector of synapses to optimize
     * @param num_neurons Number of neurons
     * @param iterations Number of optimization iterations
     */
    void optimizeForInformationFlow(std::vector<GPUSynapse>& synapses,
                                  int num_neurons, int iterations = 1000);
    
    /**
     * Optimize topology for stability
     * @param synapses Vector of synapses to optimize
     * @param num_neurons Number of neurons
     * @param target_clustering Target clustering coefficient
     */
    void optimizeForStability(std::vector<GPUSynapse>& synapses,
                            int num_neurons, float target_clustering = 0.3f);
    
    /**
     * Remove redundant connections
     * @param synapses Vector of synapses to prune
     * @param threshold Activity threshold for pruning
     */
    void pruneRedundantConnections(std::vector<GPUSynapse>& synapses,
                                 float threshold = 0.001f);
}

#endif // TOPOLOGY_GENERATOR_H