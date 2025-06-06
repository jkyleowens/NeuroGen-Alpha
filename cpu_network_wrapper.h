#pragma once
#if !defined(USE_CUDA) || !USE_CUDA
#include <vector>
#include "src/NetworkConfig.h"

struct NetworkStats {
    float avg_firing_rate{0};
    float total_spikes{0};
    float avg_weight{0};
    float reward_signal{0};
    int update_count{0};
};

void initializeNetwork();
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward);
void updateSynapticWeightsCUDA(float reward);
void cleanupNetwork();
NetworkStats getNetworkStats();
NetworkConfig getNetworkConfig();
#endif
