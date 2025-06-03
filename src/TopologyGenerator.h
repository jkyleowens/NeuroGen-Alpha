#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include "NetworkConfig.h"
#include "cuda/GPUNeuralStructures.h"
#include "GPUCorticalColumnFwd.h"

#include <vector>
#include <random>

class TopologyGenerator {
public:
    explicit TopologyGenerator(const NetworkConfig& cfg);

    void buildLocalLoops(std::vector<GPUSynapse>& synapses,
                         const std::vector<GPUCorticalColumn>& columns);

    void buildInterColumnConnections(std::vector<GPUSynapse>& synapses,
                                     const std::vector<GPUCorticalColumn>& columns,
                                     float connection_probability);

    void buildInputConnections(std::vector<GPUSynapse>& synapses,
                               int input_start, int input_end,
                               int target_start, int target_end,
                               float connection_probability);

    void shuffleConnections(std::vector<GPUSynapse>& synapses);

    void validateTopology(const std::vector<GPUSynapse>& synapses,
                          const std::vector<GPUCorticalColumn>& columns) const;

    void printTopologyStats(const std::vector<GPUSynapse>& synapses,
                            const std::vector<GPUCorticalColumn>& columns) const;

private:
    const NetworkConfig& cfg_;
    std::mt19937 rng_;

    bool isExcitatoryNeuron(int neuron_idx, const GPUCorticalColumn& column) const;
    float generateSynapticWeight(bool is_excitatory) const;
    float generateSynapticDelay() const;
    int selectRandomTarget(const GPUCorticalColumn& column) const;
};

// Preset configurations
namespace ConfigPresets {
    NetworkConfig smallNetwork();
    NetworkConfig mediumNetwork();
    NetworkConfig largeNetwork();
    NetworkConfig testNetwork();
}

#endif // TOPOLOGY_GENERATOR_H
