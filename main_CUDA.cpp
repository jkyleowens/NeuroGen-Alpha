/**
 * @file main_cuda.cpp
 * @brief CUDA-Enhanced Neural Network Demonstration
 * 
 * This program demonstrates the CUDA-accelerated biologically inspired neural network
 * with performance comparisons between CPU and GPU implementations.
 */

#include "Network_CUDA.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

/**
 * @brief Test CUDA availability and device information
 */
void testCUDAAvailability() {
    std::cout << "\n=== CUDA Availability Test ===\n";
    
    bool cuda_available = NetworkCUDAUtils::isCUDAAvailable();
    std::cout << "CUDA Available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    if (cuda_available) {
        int device_count = NetworkCUDAUtils::getCUDADeviceCount();
        std::cout << "CUDA Devices: " << device_count << std::endl;
        
        for (int i = 0; i < device_count; ++i) {
            std::cout << "  Device " << i << ": " << NetworkCUDAUtils::getCUDADeviceInfo(i) << std::endl;
        }
        
        // Test optimal configuration
        auto optimal_config = NetworkCUDAUtils::getOptimalCUDAConfig(1000, 5000);
        std::cout << "Optimal CUDA Config: " << optimal_config.toString() << std::endl;
    }
}

/**
 * @brief Performance comparison between CPU and GPU implementations
 */
void performanceComparison() {
    std::cout << "\n=== CPU vs GPU Performance Comparison ===\n";
    
    std::vector<size_t> network_sizes = {50, 100, 250, 500, 1000, 2000};
    
    std::cout << std::setw(10) << "Neurons" 
              << std::setw(15) << "CPU Time (ms)" 
              << std::setw(15) << "GPU Time (ms)" 
              << std::setw(12) << "Speedup" 
              << std::setw(15) << "Faster Method" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (size_t num_neurons : network_sizes) {
        // CPU-only configuration
        NetworkConfigCUDA cpu_config;
        cpu_config.enable_cuda = false;
        cpu_config.max_neurons = num_neurons * 2;
        cpu_config.dt = 0.01;
        cpu_config.enable_neurogenesis = false;
        cpu_config.enable_pruning = false;
        
        // GPU configuration
        NetworkConfigCUDA gpu_config = cpu_config;
        gpu_config.enable_cuda = NetworkCUDAUtils::isCUDAAvailable();
        
        try {
            // Create networks
            auto cpu_network = std::make_shared<NetworkCUDA>(cpu_config);
            auto gpu_network = std::make_shared<NetworkCUDA>(gpu_config);
            
            // Create identical network structures
            std::vector<size_t> neuron_ids_cpu, neuron_ids_gpu;
            
            for (size_t i = 0; i < num_neurons; ++i) {
                Position3D pos(i * 10.0, (i % 10) * 10.0, 0.0);
                std::string id = "neuron_" + std::to_string(i);
                
                auto neuron_cpu = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
                auto neuron_gpu = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
                
                neuron_ids_cpu.push_back(cpu_network->addNeuron(neuron_cpu, pos));
                neuron_ids_gpu.push_back(gpu_network->addNeuron(neuron_gpu, pos));
            }
            
            // Create connections
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> prob_dist(0.0, 1.0);
            
            double connection_prob = std::min(0.1, 200.0 / num_neurons); // Adaptive connection probability
            
            for (size_t i = 0; i < num_neurons; ++i) {
                for (size_t j = 0; j < num_neurons; ++j) {
                    if (i != j && prob_dist(gen) < connection_prob) {
                        double weight = 0.1 + 0.1 * prob_dist(gen);
                        cpu_network->createSynapse(neuron_ids_cpu[i], neuron_ids_cpu[j], "dendrite", 0, weight);
                        gpu_network->createSynapse(neuron_ids_gpu[i], neuron_ids_gpu[j], "dendrite", 0, weight);
                    }
                }
            }
            
            const int simulation_steps = 1000; // 10ms simulation
            const double dt = 0.01;
            
            // CPU timing
            auto cpu_start = std::chrono::high_resolution_clock::now();
            
            for (int step = 0; step < simulation_steps; ++step) {
                if (step % 100 < 10) {
                    for (size_t i = 0; i < std::min(num_neurons / 10, size_t(5)); ++i) {
                        cpu_network->injectCurrent(neuron_ids_cpu[i], 8.0);
                    }
                }
                cpu_network->step(dt);
            }
            
            auto cpu_end = std::chrono::high_resolution_clock::now();
            auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
            
            // GPU timing (if available)
            double gpu_time_ms = 0.0;
            double speedup = 0.0;
            std::string faster_method = "CPU";
            
            if (gpu_config.enable_cuda && gpu_network->isCUDAEnabled()) {
                auto gpu_start = std::chrono::high_resolution_clock::now();
                
                for (int step = 0; step < simulation_steps; ++step) {
                    if (step % 100 < 10) {
                        for (size_t i = 0; i < std::min(num_neurons / 10, size_t(5)); ++i) {
                            gpu_network->injectCurrent(neuron_ids_gpu[i], 8.0);
                        }
                    }
                    gpu_network->step(dt);
                }
                
                auto gpu_end = std::chrono::high_resolution_clock::now();
                auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
                
                gpu_time_ms = gpu_duration.count();
                
                if (gpu_time_ms > 0) {
                    speedup = static_cast<double>(cpu_duration.count()) / gpu_time_ms;
                    faster_method = (speedup > 1.0) ? "GPU" : "CPU";
                }
            } else {
                gpu_time_ms = -1; // Indicate GPU not available
                faster_method = "CPU (only)";
            }
            
            // Output results
            std::cout << std::setw(10) << num_neurons
                      << std::setw(15) << cpu_duration.count();
            
            if (gpu_time_ms >= 0) {
                std::cout << std::setw(15) << std::fixed << std::setprecision(1) << gpu_time_ms
                          << std::setw(12) << std::setprecision(2) << speedup;
            } else {
                std::cout << std::setw(15) << "N/A"
                          << std::setw(12) << "N/A";
            }
            
            std::cout << std::setw(15) << faster_method << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << std::setw(10) << num_neurons
                      << " Error: " << e.what() << std::endl;
        }
    }
}

/**
 * @brief Test CUDA-specific features
 */
void testCUDAFeatures() {
    std::cout << "\n=== CUDA-Specific Features Test ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available, skipping CUDA-specific tests" << std::endl;
        return;
    }
    
    NetworkConfigCUDA config;
    config.enable_cuda = true;
    config.max_neurons = 500;
    config.adaptive_processing = true;
    config.gpu_load_threshold = 100;
    
    auto network = std::make_shared<NetworkCUDA>(config);
    
    std::cout << "CUDA Device Info: " << network->getCUDADeviceInfo() << std::endl;
    std::cout << "CUDA Enabled: " << (network->isCUDAEnabled() ? "Yes" : "No") << std::endl;
    
    // Create a medium-sized network
    std::vector<size_t> neuron_ids;
    for (int i = 0; i < 200; ++i) {
        Position3D pos(i * 15.0, (i % 10) * 15.0, 0.0);
        std::string id = "cuda_neuron_" + std::to_string(i);
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
        neuron_ids.push_back(network->addNeuron(neuron, pos));
    }
    
    // Create connections
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    
    for (size_t i = 0; i < neuron_ids.size(); ++i) {
        for (size_t j = 0; j < neuron_ids.size(); ++j) {
            if (i != j && prob_dist(gen) < 0.05) {
                network->createSynapse(neuron_ids[i], neuron_ids[j], "dendrite", 0, 0.1 + 0.1 * prob_dist(gen));
            }
        }
    }
    
    std::cout << "Network created with " << network->getNumNeurons() << " neurons and " 
              << network->getNumSynapses() << " synapses" << std::endl;
    
    // Test GPU memory estimation
    size_t gpu_memory_needed = network->estimateGPUMemoryUsage();
    size_t gpu_memory_available = network->getAvailableGPUMemory();
    
    std::cout << "GPU Memory Needed: " << gpu_memory_needed / 1024 / 1024 << " MB" << std::endl;
    std::cout << "GPU Memory Available: " << gpu_memory_available / 1024 / 1024 << " MB" << std::endl;
    
    // Test GPU/CPU switching
    std::cout << "\nTesting adaptive GPU/CPU switching..." << std::endl;
    
    network->resetPerformanceCounters();
    
    // Run simulation with performance monitoring
    for (int step = 0; step < 500; ++step) {
        if (step % 50 < 5) {
            for (size_t i = 0; i < 10; ++i) {
                network->injectCurrent(neuron_ids[i], 10.0);
            }
        }
        
        network->step(0.01);
        
        if (step % 100 == 0) {
            auto stats = network->getPerformanceStats();
            std::cout << "Step " << step << ": GPU calls=" << stats.gpu_kernel_calls 
                      << ", CPU calls=" << stats.cpu_step_calls << std::endl;
        }
    }
    
    // Final performance statistics
    auto final_stats = network->getPerformanceStats();
    std::cout << "\nFinal Performance Statistics:" << std::endl;
    std::cout << "  Total GPU time: " << final_stats.total_gpu_time << " ms" << std::endl;
    std::cout << "  Total CPU time: " << final_stats.total_cpu_time << " ms" << std::endl;
    std::cout << "  GPU kernel calls: " << final_stats.gpu_kernel_calls << std::endl;
    std::cout << "  CPU step calls: " << final_stats.cpu_step_calls << std::endl;
    
    if (final_stats.total_cpu_time > 0 && final_stats.total_gpu_time > 0) {
        std::cout << "  GPU speedup: " << std::fixed << std::setprecision(2) 
                  << final_stats.gpu_speedup << "x" << std::endl;
    }
}

/**
 * @brief Large-scale network simulation to demonstrate CUDA benefits
 */
void largeCUDASimulation() {
    std::cout << "\n=== Large-Scale CUDA Network Simulation ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available, skipping large-scale simulation" << std::endl;
        return;
    }
    
    NetworkConfigCUDA config;
    config.enable_cuda = true;
    config.max_neurons = 5000;
    config.enable_stdp = true;
    config.enable_neuromodulation = true;
    config.adaptive_processing = false; // Force GPU usage
    config.dt = 0.01;
    
    auto network = std::make_shared<NetworkCUDA>(config);
    
    std::cout << "Creating large network with 2000 neurons..." << std::endl;
    
    // Create a large network using NetworkBuilderCUDA
    NetworkBuilderCUDA builder;
    auto large_network = builder
        .setCUDAConfig(config)
        .enableCUDA(true)
        .addNeuronPopulation(NeuronFactory::PYRAMIDAL_CORTICAL, 1500, Position3D(500.0, 500.0, 0.0), 200.0)
        .addNeuronPopulation(NeuronFactory::INTERNEURON_BASKET, 300, Position3D(200.0, 200.0, 0.0), 100.0)
        .addNeuronPopulation(NeuronFactory::SENSORY, 200, Position3D(800.0, 200.0, 0.0), 80.0)
        .addRandomConnections(0.02)
        .buildCUDA();
    
    if (!large_network->isCUDAEnabled()) {
        std::cout << "Failed to enable CUDA for large network" << std::endl;
        return;
    }
    
    std::cout << "Large network created successfully" << std::endl;
    std::cout << "  Neurons: " << large_network->getNumNeurons() << std::endl;
    std::cout << "  Synapses: " << large_network->getNumSynapses() << std::endl;
    std::cout << "  GPU Memory Usage: " << large_network->estimateGPUMemoryUsage() / 1024 / 1024 << " MB" << std::endl;
    
    // Simulation with complex activity patterns
    const int total_steps = 5000; // 50ms simulation
    const double dt = 0.01;
    
    auto sim_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Running large-scale simulation..." << std::endl;
    
    for (int step = 0; step < total_steps; ++step) {
        // Complex stimulation pattern
        if (step % 200 < 20) {
            // Burst stimulation
            for (size_t i = 0; i < std::min(large_network->getNumNeurons(), size_t(50)); ++i) {
                large_network->injectCurrent(i, 12.0 + 3.0 * sin(step * 0.1));
            }
        } else if (step % 500 == 250) {
            // Reward signal
            large_network->releaseNeuromodulator("dopamine", 1.0);
        } else if (step % 100 < 5) {
            // Background activity
            for (size_t i = 1500; i < std::min(large_network->getNumNeurons(), size_t(1600)); ++i) {
                large_network->injectCurrent(i, 5.0);
            }
        }
        
        large_network->step(dt);
        
        // Progress reporting
        if (step % 1000 == 0) {
            auto current_stats = large_network->calculateNetworkStats(50.0);
            std::cout << "  Step " << step << ": Firing rate = " 
                      << std::fixed << std::setprecision(1) << current_stats.mean_firing_rate 
                      << " Hz, Active neurons = " << current_stats.active_neurons << std::endl;
        }
    }
    
    auto sim_end = std::chrono::high_resolution_clock::now();
    auto sim_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sim_end - sim_start);
    
    // Final analysis
    auto final_stats = large_network->calculateNetworkStats(100.0);
    auto perf_stats = large_network->getPerformanceStats();
    
    std::cout << "\nSimulation completed in " << sim_duration.count() << " ms" << std::endl;
    std::cout << "\nFinal Network Statistics:" << std::endl;
    std::cout << "  Active neurons: " << final_stats.active_neurons << std::endl;
    std::cout << "  Total synapses: " << final_stats.total_synapses << std::endl;
    std::cout << "  Mean firing rate: " << final_stats.mean_firing_rate << " Hz" << std::endl;
    std::cout << "  Network synchrony: " << final_stats.network_synchrony << std::endl;
    std::cout << "  Mean connectivity: " << final_stats.mean_connectivity << std::endl;
    
    std::cout << "\nGPU Performance:" << std::endl;
    std::cout << "  GPU time: " << perf_stats.total_gpu_time << " ms" << std::endl;
    std::cout << "  GPU kernel calls: " << perf_stats.gpu_kernel_calls << std::endl;
    
    double simulated_time = total_steps * dt; // ms
    double real_time = sim_duration.count(); // ms
    double real_time_ratio = simulated_time / real_time;
    
    std::cout << "  Real-time performance: " << std::fixed << std::setprecision(2) 
              << real_time_ratio << "x real-time" << std::endl;
    
    if (real_time_ratio >= 1.0) {
        std::cout << "  Network can run in real-time!" << std::endl;
    }
}

/**
 * @brief Memory optimization test
 */
void testMemoryOptimization() {
    std::cout << "\n=== Memory Optimization Test ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available, skipping memory tests" << std::endl;
        return;
    }
    
    // Test memory estimation
    std::vector<size_t> test_sizes = {100, 500, 1000, 2000, 5000};
    
    std::cout << std::setw(10) << "Neurons" 
              << std::setw(15) << "Est. Memory (MB)" 
              << std::setw(15) << "Can Fit on GPU" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (size_t neurons : test_sizes) {
        size_t synapses = neurons * 10; // Estimate 10 synapses per neuron
        bool can_fit = NetworkCUDAUtils::canFitOnGPU(neurons, synapses);
        
        NetworkConfigCUDA config;
        config.max_neurons = neurons;
        auto network = std::make_shared<NetworkCUDA>(config);
        
        // Create minimal network for estimation
        for (size_t i = 0; i < std::min(neurons, size_t(10)); ++i) {
            Position3D pos(i * 10.0, 0.0, 0.0);
            auto neuron = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, "test", pos);
            network->addNeuron(neuron, pos);
        }
        
        size_t estimated_memory = network->estimateGPUMemoryUsage();
        
        std::cout << std::setw(10) << neurons
                  << std::setw(15) << estimated_memory / 1024 / 1024
                  << std::setw(15) << (can_fit ? "Yes" : "No") << std::endl;
    }
    
    // Test optimal batch size calculation
    size_t available_memory = NetworkCUDAUtils::isCUDAAvailable() ? 
        std::make_shared<NetworkCUDA>()->getAvailableGPUMemory() : 0;
    
    if (available_memory > 0) {
        std::cout << "\nAvailable GPU Memory: " << available_memory / 1024 / 1024 << " MB" << std::endl;
        
        size_t optimal_batch = NetworkCUDAUtils::calculateOptimalBatchSize(available_memory, 10000);
        std::cout << "Optimal batch size for 10k neurons: " << optimal_batch << std::endl;
    }
}

/**
 * @brief CUDA error handling test
 */
void testErrorHandling() {
    std::cout << "\n=== CUDA Error Handling Test ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available, skipping error handling tests" << std::endl;
        return;
    }
    
    // Test graceful fallback to CPU
    NetworkConfigCUDA config;
    config.enable_cuda = true;
    config.max_neurons = 1000000; // Intentionally large to potentially cause memory issues
    
    try {
        auto network = std::make_shared<NetworkCUDA>(config);
        
        if (network->isCUDAEnabled()) {
            std::cout << "Large network configuration accepted" << std::endl;
        } else {
            std::cout << "Network gracefully fell back to CPU mode" << std::endl;
        }
        
        // Test with reasonable size
        config.max_neurons = 100;
        auto small_network = std::make_shared<NetworkCUDA>(config);
        
        std::cout << "Small network CUDA enabled: " << small_network->isCUDAEnabled() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Caught exception (expected): " << e.what() << std::endl;
    }
    
    // Test CUDA error utilities
    NetworkCUDAUtils::clearCUDAErrors();
    std::string last_error = NetworkCUDAUtils::getLastCUDAError();
    std::cout << "Last CUDA error: " << last_error << std::endl;
}

int main() {
    std::cout << "CUDA-Enhanced Biologically Inspired Neural Network - Test Suite\n";
    std::cout << "==============================================================\n";
    
    try {
        // Test CUDA availability
        testCUDAAvailability();
        
        // Performance comparisons
        performanceComparison();
        
        // CUDA-specific features
        testCUDAFeatures();
        
        // Large-scale simulation (only if CUDA is available)
        largeCUDASimulation();
        
        // Memory optimization
        testMemoryOptimization();
        
        // Error handling
        testErrorHandling();
        
        std::cout << "\n=== All CUDA Tests Completed ===\n";
        
        // Summary and recommendations
        if (NetworkCUDAUtils::isCUDAAvailable()) {
            auto recommended_config = NetworkCUDAUtils::recommendCUDASettings();
            std::cout << "\nRecommended CUDA Configuration:\n";
            std::cout << recommended_config.toString() << std::endl;
            
            std::cout << "\nFor optimal performance:\n";
            std::cout << "- Use GPU for networks with >100 neurons\n";
            std::cout << "- Enable adaptive processing for mixed workloads\n";
            std::cout << "- Use pinned memory for faster transfers\n";
            std::cout << "- Monitor GPU memory usage for large networks\n";
        } else {
            std::cout << "\nCUDA not available on this system.\n";
            std::cout << "The network will run in CPU-only mode.\n";
            std::cout << "Consider installing CUDA drivers and toolkit for GPU acceleration.\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during CUDA testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

/*
CUDA COMPILATION INSTRUCTIONS:
==============================

To compile with CUDA support:

1. Install CUDA Toolkit (11.0 or later recommended)
2. Install compatible compiler (GCC 7-9 for CUDA 11.x)

Compilation commands:

# With CMake (recommended):
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
make -j4

# Or with nvcc directly:
nvcc -std=c++14 -O3 -o network_cuda_demo \
     main_cuda.cpp Network_CUDA.cpp NetworkCUDA.cu Network.cpp Neuron.cpp \
     -lcudart -lcublas -lcurand

# For debug build:
nvcc -std=c++14 -g -G -o network_cuda_debug \
     main_cuda.cpp Network_CUDA.cpp NetworkCUDA.cu Network.cpp Neuron.cpp \
     -lcudart -lcublas -lcurand

Dependencies:
- CUDA Toolkit 11.0+
- C++14 compatible compiler
- NVIDIA GPU with Compute Capability 3.5+

Runtime Requirements:
- NVIDIA GPU driver
- CUDA runtime libraries

Performance Notes:
- GPU acceleration most beneficial for >100 neurons
- Memory bandwidth typically limits performance
- Real-time performance achievable for networks up to 5000 neurons
- Adaptive CPU/GPU switching optimizes performance across problem sizes

Memory Requirements:
- ~1KB per neuron on GPU
- ~100 bytes per synapse on GPU
- Recommend 2GB+ GPU memory for large networks

Expected Speedup:
- 2-10x for medium networks (100-1000 neurons)
- 5-20x for large networks (1000+ neurons)
- Depends on GPU model and network connectivity
*/