#include "include/NeuroGen/NetworkConfig.h"
#include <iostream>

int main() {
    std::cout << "Testing NetworkConfig..." << std::endl;
    
    NetworkConfig config;
    config.input_size = 32;
    config.hidden_size = 128;
    config.output_size = 8;
    
    std::cout << "Config created successfully!" << std::endl;
    std::cout << "Input size: " << config.input_size << std::endl;
    
    return 0;
}
