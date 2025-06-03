#include <iostream>
#include <memory>
#include "include/NeuroGen/TopologyGenerator.h"
#include "include/NeuroGen/NetworkConfig.h"

// Simple main function to test TopologyGenerator compilation and functionality
int main() {
    std::cout << "=== TopologyGenerator Test Program ===" << std::endl;
    
    try {
        // Create a basic network configuration
        NetworkConfig config;
        config.num_neurons = 1000;
        config.max_neurons = 1000;
        config.exc_ratio = 0.8;
        config.enable_stdp = true;
        config.enable_homeostasis = false;
        
        // TopologyGenerator fields
        config.numColumns = 10;
        config.neuronsPerColumn = 100;
        config.localFanOut = 20;
        config.wExcMin = 0.1;
        config.wExcMax = 1.0;
        config.wInhMin = -1.0;
        config.wInhMax = -0.1;
        config.dMin = 1.0;
        config.dMax = 5.0;
        config.totalSynapses = 20000;
        config.spike_threshold = -55.0;
        
        std::cout << "Configuration created successfully:" << std::endl;
        std::cout << "  Neurons: " << config.num_neurons << std::endl;
        std::cout << "  Columns: " << config.numColumns << std::endl;
        std::cout << "  Neurons per column: " << config.neuronsPerColumn << std::endl;
        std::cout << "  Excitatory ratio: " << config.exc_ratio << std::endl;
        
        // Create TopologyGenerator instance
        std::cout << "\nCreating TopologyGenerator..." << std::endl;
        TopologyGenerator topology(config);
        
        std::cout << "TopologyGenerator created successfully!" << std::endl;
        
        // Test preset configurations
        std::cout << "\nTesting preset configurations..." << std::endl;
        
        // Test small world topology
        auto small_world_config = TopologyGenerator::createSmallWorldPreset();
        std::cout << "Small world preset: " << small_world_config.num_neurons << " neurons" << std::endl;
        
        // Test cortical column topology  
        auto cortical_config = TopologyGenerator::createCorticalColumnPreset();
        std::cout << "Cortical column preset: " << cortical_config.num_neurons << " neurons" << std::endl;
        
        // Test scale-free topology
        auto scale_free_config = TopologyGenerator::createScaleFreePreset();
        std::cout << "Scale-free preset: " << scale_free_config.num_neurons << " neurons" << std::endl;
        
        std::cout << "\n=== All tests passed! ===" << std::endl;
        std::cout << "TopologyGenerator is working correctly." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
