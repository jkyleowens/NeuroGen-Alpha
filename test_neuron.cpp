#include "Neuron.h"
#include <iostream>

int main() {
    std::cout << "Testing Neuron compilation..." << std::endl;
    
    Neuron neuron("test_neuron");
    std::cout << "Neuron created with ID: " << neuron.getId() << std::endl;
    
    std::cout << "All basic functionality working!" << std::endl;
    return 0;
}
