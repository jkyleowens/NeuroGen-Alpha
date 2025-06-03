#include <iostream>
#include "NetworkConfig.h"

int main() {
    std::cout << "Testing basic NetworkConfig..." << std::endl;
    
    NetworkConfig config;
    config.numColumns = 2;
    config.neuronsPerColumn = 100;
    config.localFanOut = 10;
    config.exc_ratio = 0.8f;
    config.wExcMin = 0.1f;
    config.wExcMax = 1.0f;
    config.wInhMin = 0.1f;
    config.wInhMax = 1.0f;
    config.dMin = 1.0f;
    config.dMax = 10.0f;
    
    std::cout << "Basic config created" << std::endl;
    
    try {
        config.finalizeConfig();
        std::cout << "Config finalized successfully" << std::endl;
        std::cout << "Total synapses: " << config.totalSynapses << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
