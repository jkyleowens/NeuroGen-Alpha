#include "Network.h"
#include <iostream>

int main() {
    std::cout << "Creating basic network..." << std::endl;
    
    NetworkConfig config;
    Network network(config);
    
    std::cout << "Network created successfully!" << std::endl;
    std::cout << "Initial neuron count: " << network.getNeuronCount() << std::endl;
    
    // Add a few neurons
    network.addNeuron(Position3D(0, 0, 0));
    network.addNeuron(Position3D(1, 0, 0));
    network.addNeuron(Position3D(2, 0, 0));
    
    std::cout << "Added neurons. Total count: " << network.getNeuronCount() << std::endl;
    
    // Run a brief simulation
    for (int i = 0; i < 10; ++i) {
        network.step(0.1);
    }
    
    std::cout << "Simulation completed successfully!" << std::endl;
    return 0;
}
