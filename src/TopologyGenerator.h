#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <string>
#include <map>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/NetworkConfig.h>

// Forward declaration
class Network; 

// Defines a specific rule for connecting one group of neurons to another.
struct ConnectionRule {
    enum class Type {
        PROBABILISTIC,      // Simple probability-based connection.
        DISTANCE_DECAY,     // Connection probability decays with distance.
        FIXED_IN_DEGREE,    // Each target neuron receives a fixed number of inputs.
        FIXED_OUT_DEGREE    // Each source neuron sends a fixed number of outputs.
    };

    std::string source_pop_key; // Key for the source population of neurons
    std::string target_pop_key; // Key for the target population

    Type type = Type::PROBABILISTIC;
    float probability = 0.1f;         // For PROBABILISTIC type
    float max_distance = 150.0f;      // For DISTANCE_DECAY type
    float decay_constant = 50.0f;     // For DISTANCE_DECAY type
    int degree = 10;                  // For FIXED_IN_DEGREE / FIXED_OUT_DEGREE

    // Synaptic properties for this rule
    float weight_mean = 0.1f;
    float weight_std_dev = 0.05f;
    float delay_min = 1.0f;           // ms
    float delay_max = 5.0f;           // ms
    int receptor_type = RECEPTOR_AMPA;
};

// A more abstract representation of a neuron for topology generation before full creation.
struct NeuronModel {
    int id;
    float x, y, z;
    std::string population_key;
};

class TopologyGenerator {
public:
    TopologyGenerator(const NetworkConfig& config);

    // Main function to generate synapses based on a set of neuron models and rules.
    std::vector<GPUSynapse> generate(
        const std::map<std::string, std::vector<NeuronModel>>& populations,
        const std::vector<ConnectionRule>& rules
    );

private:
    // Spatial hashing grid for efficient distance-based connection generation.
    using SpatialGrid = std::vector<std::vector<std::vector<std::vector<int>>>>;

    void initializeSpatialGrid(const std::vector<NeuronModel>& all_neurons);
    std::vector<int> getNearbyNeurons(const NeuronModel& neuron, float radius);

    // Rule-specific generation methods
    void applyDistanceDecayRule(
        const ConnectionRule& rule,
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        std::vector<GPUSynapse>& synapses
    );

    void applyProbabilisticRule(
        const ConnectionRule& rule,
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        std::vector<GPUSynapse>& synapses
    );
    
    // Helper to create a single synapse
    GPUSynapse createSynapse(int pre_id, int post_id, int receptor, float weight, float delay);

    const NetworkConfig& config_;
    std::mt19937 rng_;
    SpatialGrid spatial_grid_;
    std::vector<NeuronModel> grid_neurons_;
    float grid_bin_size_ = 50.0f; // Spatial bin size in um
};

#endif // TOPOLOGY_GENERATOR_H