// Simple test for network functionality
#include <iostream>
#include <vector>
#include "src/cuda/NetworkCUDA.cuh"

int main() {
    std::cout << "Testing CUDA Neural Network..." << std::endl;
    
    // Initialize network
    initializeNetwork();
    
    // Create sample input (60 features)
    std::vector<float> test_input(60);
    for (int i = 0; i < 60; ++i) {
        test_input[i] = 0.1f * i / 60.0f; // Simple ramp
    }
    
    // Test forward pass
    std::vector<float> output = forwardCUDA(test_input, 0.0f);
    
    std::cout << "Network output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Test learning update
    updateSynapticWeightsCUDA(1.0f); // Positive reward
    
    // Test again to see if output changes
    output = forwardCUDA(test_input, 1.0f);
    
    std::cout << "Output after learning: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cleanupNetwork();
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
