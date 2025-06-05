#include "TopologyGenerator.h"
#include "NetworkPresets.h"
#include <iostream>
#include <vector>

int main() {
    try {
        // Test with balanced default configuration
        NetworkConfig config = NetworkPresets::balanced_default();
        
        // Initialize the configuration
        config.finalizeConfig();
        
        std::cout << "Creating TopologyGenerator..." << std::endl;
        TopologyGenerator tg(config);
        
        // Create some test columns
        std::vector<GPUCorticalColumn> columns(config.numColumns);
        for (int i = 0; i < config.numColumns; ++i) {
            columns[i].neuron_start = i * config.neuronsPerColumn;
            columns[i].neuron_end = (i + 1) * config.neuronsPerColumn;
            columns[i].column_id = i;
            columns[i].neuron_count = config.neuronsPerColumn;
            columns[i].excitatory_count = static_cast<int>(config.exc_ratio * config.neuronsPerColumn);
            columns[i].inhibitory_count = config.neuronsPerColumn - columns[i].excitatory_count;
        }
        
        // Test building local loops
        std::vector<GPUSynapse> synapses;
        std::cout << "Building local loops..." << std::endl;
        tg.buildLocalLoops(synapses, columns);

        if (synapses.size() != config.totalSynapses) {
            std::cerr << "Synapse count mismatch: expected "
                      << config.totalSynapses << ", got "
                      << synapses.size() << std::endl;
            return 1;
        }

        std::cout << "Created " << synapses.size() << " synapses" << std::endl;
        
        // Test validation
        std::cout << "Validating topology..." << std::endl;
        tg.validateTopology(synapses, columns);
        
        // Print stats
        std::cout << "Printing topology stats..." << std::endl;
        tg.printTopologyStats(synapses, columns);
        
        std::cout << "TopologyGenerator test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
