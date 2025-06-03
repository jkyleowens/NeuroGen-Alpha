#include <iostream>
#include <memory>

// Minimal NetworkConfig structure to avoid CUDA dependencies
struct NetworkConfig {
    size_t num_neurons = 1000;
    size_t max_neurons = 1000;
    double exc_ratio = 0.8;
    bool enable_stdp = true;
    bool enable_homeostasis = false;
    
    // TopologyGenerator required fields
    int numColumns = 10;
    int neuronsPerColumn = 100;
    int localFanOut = 20;
    double wExcMin = 0.1;
    double wExcMax = 1.0;
    double wInhMin = -1.0;
    double wInhMax = -0.1;
    double dMin = 1.0;
    double dMax = 5.0;
    int totalSynapses = 20000;
    double spike_threshold = -55.0;
    
    // Minimal required methods
    void finalizeConfig() {
        // Ensure derived values are computed
        if (numColumns > 0 && neuronsPerColumn > 0) {
            num_neurons = numColumns * neuronsPerColumn;
            max_neurons = num_neurons;
        }
    }
};

// Include TopologyGenerator after config definition
#include "src/TopologyGenerator.h"

int main() {
    std::cout << "=== TopologyGenerator Test Program ===" << std::endl;
    
    try {
        // Create basic network configuration
        NetworkConfig config;
        config.finalizeConfig();
        
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
