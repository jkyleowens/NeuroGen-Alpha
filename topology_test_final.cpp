#include <iostream>
#include "TopologyGenerator.h"
#include "NetworkPresets.h"

int main() {
    try {
        std::cout << "=== TopologyGenerator Test ===" << std::endl;
        
        // Create a simple config
        NetworkConfig config = NetworkPresets::balanced_default();
        std::cout << "✓ Config created" << std::endl;
        
        // Initialize TopologyGenerator
        TopologyGenerator tg(config);
        std::cout << "✓ TopologyGenerator created" << std::endl;
        
        // Create test columns
        std::vector<GPUCorticalColumn> columns(config.numColumns);
        for (size_t i = 0; i < columns.size(); ++i) {
            columns[i].neuron_start = static_cast<int>(i * config.neuronsPerColumn);
            columns[i].neuron_end = static_cast<int>((i + 1) * config.neuronsPerColumn);
            columns[i].column_id = static_cast<int>(i);
            columns[i].neuron_count = config.neuronsPerColumn;
            columns[i].excitatory_count = static_cast<int>(config.exc_ratio * config.neuronsPerColumn);
            columns[i].inhibitory_count = config.neuronsPerColumn - columns[i].excitatory_count;
            columns[i].active = true;
        }
        std::cout << "✓ Columns created: " << columns.size() << std::endl;
        
        // Test building connections
        std::vector<GPUSynapse> synapses;
        tg.buildLocalLoops(synapses, columns);
        std::cout << "✓ Local loops built: " << synapses.size() << " synapses" << std::endl;
        
        // Test validation
        tg.validateTopology(synapses, columns);
        std::cout << "✓ Topology validation passed" << std::endl;
        
        // Print stats
        tg.printTopologyStats(synapses, columns);
        
        std::cout << "\n=== TopologyGenerator Test PASSED ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception" << std::endl;
        return 1;
    }
}
