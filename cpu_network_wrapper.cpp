#include "NetworkCPU.h"
#include "src/NetworkConfig.h"
#include <vector>
#include <memory>

#if !defined(USE_CUDA) || !USE_CUDA
static std::unique_ptr<NetworkCPU> g_cpu_net;

void initializeNetwork() {
    NetworkConfig cfg;
    g_cpu_net = std::make_unique<NetworkCPU>(cfg);
}

std::vector<float> forwardCUDA(const std::vector<float>& input, float reward) {
    if (!g_cpu_net)
        initializeNetwork();
    return g_cpu_net->forward(input, reward);
}

void updateSynapticWeightsCUDA(float reward) {
    if (g_cpu_net)
        g_cpu_net->updateWeights(reward);
}

void cleanupNetwork() {
    if (g_cpu_net) {
        g_cpu_net->cleanup();
        g_cpu_net.reset();
    }
}

NetworkStats getNetworkStats() {
    NetworkStats stats{};
    return stats;
}

NetworkConfig getNetworkConfig() {
    NetworkConfig cfg;
    return cfg;
}
#endif
