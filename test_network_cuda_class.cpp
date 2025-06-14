#include <iostream>
#include <NeuroGen/cuda/NetworkCUDA.cuh>

int main() {
    std::cout << "Testing NetworkCUDA compilation..." << std::endl;
    
    try {
        // Create a basic network config
        NetworkConfig config;
        config.numColumns = 4;
        config.neuronsPerColumn = 10;
        config.localFanOut = 5;
        
        // Create NetworkCUDA instance
        NetworkCUDA network(config);
        
        std::cout << "NetworkCUDA created successfully!" << std::endl;
        std::cout << "Number of neurons: " << network.getNumNeurons() << std::endl;
        std::cout << "Number of synapses: " << network.getNumSynapses() << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
