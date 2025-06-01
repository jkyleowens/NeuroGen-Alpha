#pragma once
#include "GPUNeuralStructures.h"
#include "CorticalColumn.h"
#include "NetworkConfig.h"          // (add ratio + fan-in params here)
#include <vector>
#include <random>

/*  Utility that fills pre-allocated host vectors with intra-column synapses.  */
class TopologyGenerator
{
public:
    explicit TopologyGenerator(const NetworkConfig& cfg);

    /* Fills `host_synapses` with *exactly* cfg.totalSynapses elements.        */
    void buildLocalLoops(std::vector<GPUSynapse>& host_synapses,
                         const std::vector<GPUCorticalColumn>& columns);

private:
    const NetworkConfig& cfg_;
    std::mt19937                     rng_;
};
