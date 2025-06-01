/**
 * @file test_cuda.cpp
 * @brief Test program for CUDA neural network functionality
 */

#include "Network_CUDA.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "CUDA Neural Network Test" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Test CUDA availability detection
    std::cout << "Checking CUDA availability..." << std::endl;
    bool cuda_available = NetworkCUDAUtils::isCUDAAvailable();
    std::cout << "CUDA Available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    if (cuda_available) {
        std::cout << "CUDA Device Count: " << NetworkCUDAUtils::getCUDADeviceCount() << std::endl;
        std::cout << "Device 0 Info: " << NetworkCUDAUtils::getCUDADeviceInfo(0) << std::endl;
    }
    
    // Create CUDA-enhanced network configuration
    NetworkConfigCUDA config;
    config.max_neurons = 100;
    config.network_width = 100.0;
    config.network_height = 100.0;
    config.network_depth = 100.0;
    config.enable_cuda = cuda_available; // Only enable if available
    config.adaptive_processing = true;
    config.gpu_load_threshold = 10;
    
    std::cout << "\nCreating NetworkCUDA with config:" << std::endl;
    std::cout << config.toString() << std::endl;
    
    try {
        // Create CUDA-enhanced network
        auto network = std::make_shared<NetworkCUDA>(config);
        
        std::cout << "\nNetwork created successfully" << std::endl;
        std::cout << "CUDA Enabled: " << (network->isCUDAEnabled() ? "Yes" : "No") << std::endl;
        
        if (network->isCUDAEnabled()) {
            std::cout << "CUDA Device Info: " << network->getCUDADeviceInfo() << std::endl;
        }
        
        // Test basic network operations
        std::cout << "\nTesting basic network operations..." << std::endl;
        
        // Add some neurons using NetworkBuilder pattern
        NetworkBuilderCUDA builder;
        builder.setCUDAConfig(config)
               .addNeuronPopulation(NeuronFactory::NeuronType::PYRAMIDAL, 20, 
                                   Position3D(0, 0, 0), 50.0)
               .addNeuronPopulation(NeuronFactory::NeuronType::INTERNEURON, 5,
                                   Position3D(50, 50, 50), 25.0)
               .addRandomConnections(0.1)
               .enableCUDA(cuda_available);
        
        auto built_network = builder.buildCUDA();
        
        std::cout << "Built network with " << built_network->getNumNeurons() << " neurons" << std::endl;
        std::cout << "Built network with " << built_network->getNumSynapses() << " synapses" << std::endl;
        
        // Test simulation step
        std::cout << "\nTesting simulation step..." << std::endl;
        
        // Inject some current to make neurons spike
        built_network->injectCurrent(0, 10.0);
        built_network->injectCurrent(5, 8.0);
        
        // Run simulation for a few steps
        for (int i = 0; i < 10; ++i) {
            built_network->step(0.1); // 0.1ms timestep
        }
        
        std::cout << "Simulation completed successfully" << std::endl;
        
        // Test performance statistics
        if (built_network->isCUDAEnabled()) {
            auto stats = built_network->getPerformanceStats();
            std::cout << "\nPerformance Statistics:" << std::endl;
            std::cout << "GPU Time: " << stats.total_gpu_time << " ms" << std::endl;
            std::cout << "CPU Time: " << stats.total_cpu_time << " ms" << std::endl;
            std::cout << "GPU Kernel Calls: " << stats.gpu_kernel_calls << std::endl;
            std::cout << "CPU Step Calls: " << stats.cpu_step_calls << std::endl;
            
            if (stats.total_cpu_time > 0) {
                std::cout << "GPU Speedup: " << stats.gpu_speedup << "x" << std::endl;
            }
        }
        
        // Test network statistics
        auto network_stats = built_network->calculateNetworkStats(10.0);
        std::cout << "\nNetwork Statistics:" << std::endl;
        std::cout << "Average Firing Rate: " << network_stats.average_firing_rate << " Hz" << std::endl;
        std::cout << "Synchrony Index: " << network_stats.synchrony_index << std::endl;
        std::cout << "Connection Density: " << network_stats.connection_density << std::endl;
        std::cout << "Average Weight: " << network_stats.average_weight << std::endl;
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during CUDA network test: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
