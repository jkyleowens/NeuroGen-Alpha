/**
 * @file main_1024_CUDA.cpp
 * @brief Optimized CUDA Neural Network Test for 1024-Neuron Networks
 * 
 * This program is specifically designed to test and optimize CUDA acceleration
 * for large neural networks with up to 1024 neurons. It focuses on real-world
 * performance testing and optimization of CUDA kernels.
 */

#include "Network_CUDA.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <memory>

// Global configuration for 1024-neuron testing
constexpr size_t TARGET_NEURONS = 1024;
constexpr size_t MAX_NEURONS = 1024;
constexpr size_t SIMULATION_STEPS = 10000;  // 100ms at 0.01ms timestep
constexpr double TIMESTEP = 0.01;  // ms
constexpr double SIMULATION_DURATION = SIMULATION_STEPS * TIMESTEP;  // ms

/**
 * @brief Performance metrics for detailed analysis
 */
struct DetailedPerformanceMetrics {
    double total_simulation_time_ms;
    double average_step_time_ms;
    double gpu_utilization_percent;
    double memory_bandwidth_gb_s;
    double kernel_efficiency_percent;
    size_t total_spikes_generated;
    double firing_rate_hz;
    double network_synchrony;
    size_t gpu_memory_used_mb;
    size_t cpu_fallback_count;
    double cuda_overhead_ms;
    
    void print() const {
        std::cout << "\n=== Detailed Performance Metrics ===\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total Simulation Time: " << total_simulation_time_ms << " ms\n";
        std::cout << "Average Step Time: " << average_step_time_ms << " ms\n";
        std::cout << "Real-time Factor: " << (TIMESTEP / average_step_time_ms) << "x\n";
        std::cout << "GPU Utilization: " << gpu_utilization_percent << "%\n";
        std::cout << "Memory Bandwidth: " << memory_bandwidth_gb_s << " GB/s\n";
        std::cout << "Kernel Efficiency: " << kernel_efficiency_percent << "%\n";
        std::cout << "Total Spikes: " << total_spikes_generated << "\n";
        std::cout << "Network Firing Rate: " << firing_rate_hz << " Hz\n";
        std::cout << "Network Synchrony: " << network_synchrony << "\n";
        std::cout << "GPU Memory Used: " << gpu_memory_used_mb << " MB\n";
        std::cout << "CPU Fallback Count: " << cpu_fallback_count << "\n";
        std::cout << "CUDA Overhead: " << cuda_overhead_ms << " ms\n";
    }
};

/**
 * @brief Create an optimized 1024-neuron network configuration
 */
NetworkConfigCUDA createOptimal1024Config() {
    NetworkConfigCUDA config;
    
    // Core settings
    config.enable_cuda = true;
    config.max_neurons = MAX_NEURONS;
    config.dt = TIMESTEP;
    
    // CUDA optimization settings
    config.adaptive_processing = false;  // Force GPU for consistent testing
    config.gpu_load_threshold = 50;      // Use GPU for 50+ neurons
    config.use_pinned_memory = true;     // Enable pinned memory for faster transfers
    config.threads_per_block = 256;      // Optimal threads per block
    config.cuda_device_id = 0;           // Use first CUDA device
    
    // Network dynamics
    config.enable_stdp = true;
    config.enable_neuromodulation = true;
    config.enable_neurogenesis = false;  // Disable for consistent neuron count
    config.enable_pruning = false;       // Disable for consistent connectivity
    
    // Performance tuning
    config.force_gpu_sync = false;       // Async operations for better performance
    config.async_memory_transfer = true; // Enable async memory transfers
    config.gpu_memory_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
    
    return config;
}

/**
 * @brief Create a realistic 1024-neuron cortical microcircuit
 */
std::shared_ptr<NetworkCUDA> create1024NeuronNetwork() {
    std::cout << "Creating optimized 1024-neuron cortical microcircuit...\n";
    
    auto config = createOptimal1024Config();
    auto network = std::make_shared<NetworkCUDA>(config);
    
    if (!network->isCUDAEnabled()) {
        std::cout << "WARNING: CUDA not enabled, falling back to CPU mode\n";
    }
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> pos_dist(-500.0, 500.0);
    std::uniform_real_distribution<> z_dist(0.0, 1000.0);
    
    std::vector<size_t> excitatory_neurons, inhibitory_neurons;
    
    // Create 80% excitatory (pyramidal) neurons
    for (size_t i = 0; i < 819; ++i) {  // 80% of 1024
        Position3D pos(pos_dist(gen), pos_dist(gen), z_dist(gen));
        std::string id = "exc_" + std::to_string(i);
        
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
        size_t neuron_id = network->addNeuron(neuron, pos);
        excitatory_neurons.push_back(neuron_id);
    }
    
    // Create 20% inhibitory (interneuron) neurons
    for (size_t i = 0; i < 205; ++i) {  // 20% of 1024
        Position3D pos(pos_dist(gen), pos_dist(gen), z_dist(gen));
        std::string id = "inh_" + std::to_string(i);
        
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::INTERNEURON_BASKET, id, pos);
        size_t neuron_id = network->addNeuron(neuron, pos);
        inhibitory_neurons.push_back(neuron_id);
    }
    
    std::cout << "Created " << excitatory_neurons.size() << " excitatory and " 
              << inhibitory_neurons.size() << " inhibitory neurons\n";
    
    // Create realistic connectivity
    std::uniform_real_distribution<> conn_prob(0.0, 1.0);
    std::normal_distribution<> weight_dist(0.1, 0.02);
    std::uniform_int_distribution<> delay_dist(1, 5);
    
    size_t total_connections = 0;
    
    // Excitatory to excitatory connections (local connectivity, 15% probability)
    for (size_t i = 0; i < excitatory_neurons.size(); ++i) {
        for (size_t j = 0; j < excitatory_neurons.size(); ++j) {
            if (i != j && conn_prob(gen) < 0.15) {
                double weight = std::abs(weight_dist(gen));
                int delay = delay_dist(gen);
                network->createSynapse(excitatory_neurons[i], excitatory_neurons[j], 
                                     "dendrite", delay, weight);
                total_connections++;
            }
        }
    }
    
    // Excitatory to inhibitory connections (25% probability)
    for (size_t i = 0; i < excitatory_neurons.size(); ++i) {
        for (size_t j = 0; j < inhibitory_neurons.size(); ++j) {
            if (conn_prob(gen) < 0.25) {
                double weight = std::abs(weight_dist(gen)) * 1.2;  // Stronger weights
                int delay = delay_dist(gen);
                network->createSynapse(excitatory_neurons[i], inhibitory_neurons[j], 
                                     "dendrite", delay, weight);
                total_connections++;
            }
        }
    }
    
    // Inhibitory to excitatory connections (35% probability, negative weights)
    for (size_t i = 0; i < inhibitory_neurons.size(); ++i) {
        for (size_t j = 0; j < excitatory_neurons.size(); ++j) {
            if (conn_prob(gen) < 0.35) {
                double weight = -std::abs(weight_dist(gen)) * 2.0;  // Inhibitory weights
                int delay = delay_dist(gen);
                network->createSynapse(inhibitory_neurons[i], excitatory_neurons[j], 
                                     "dendrite", delay, weight);
                total_connections++;
            }
        }
    }
    
    // Inhibitory to inhibitory connections (15% probability, negative weights)
    for (size_t i = 0; i < inhibitory_neurons.size(); ++i) {
        for (size_t j = 0; j < inhibitory_neurons.size(); ++j) {
            if (i != j && conn_prob(gen) < 0.15) {
                double weight = -std::abs(weight_dist(gen)) * 1.5;
                int delay = delay_dist(gen);
                network->createSynapse(inhibitory_neurons[i], inhibitory_neurons[j], 
                                     "dendrite", delay, weight);
                total_connections++;
            }
        }
    }
    
    std::cout << "Created " << total_connections << " synaptic connections\n";
    std::cout << "Average connectivity: " << (double)total_connections / TARGET_NEURONS << " connections/neuron\n";
    
    return network;
}

/**
 * @brief Run comprehensive performance benchmark
 */
DetailedPerformanceMetrics runPerformanceBenchmark(std::shared_ptr<NetworkCUDA>& network) {
    std::cout << "\n=== Running Performance Benchmark ===\n";
    
    DetailedPerformanceMetrics metrics = {};
    
    // Reset performance counters
    network->resetPerformanceCounters();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> stim_current(8.0, 15.0);
    std::uniform_int_distribution<> stim_neuron(0, TARGET_NEURONS - 1);
    
    // Warmup phase
    std::cout << "Warmup phase (1000 steps)...\n";
    for (int step = 0; step < 1000; ++step) {
        if (step % 100 == 0) {
            // Inject stimulus
            for (int i = 0; i < 10; ++i) {
                size_t neuron_id = stim_neuron(gen);
                network->injectCurrent(neuron_id, stim_current(gen));
            }
        }
        network->step(TIMESTEP);
    }
    
    // Main benchmark
    std::cout << "Running main benchmark (" << SIMULATION_STEPS << " steps)...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    
    size_t spike_count = 0;
    std::vector<double> step_times;
    step_times.reserve(SIMULATION_STEPS);
    
    for (int step = 0; step < SIMULATION_STEPS; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Complex stimulation pattern
        if (step % 500 == 0) {
            // Burst stimulation every 5ms
            for (int i = 0; i < 20; ++i) {
                size_t neuron_id = stim_neuron(gen);
                network->injectCurrent(neuron_id, stim_current(gen));
            }
        } else if (step % 1000 == 500) {
            // Background stimulation
            for (int i = 0; i < 5; ++i) {
                size_t neuron_id = stim_neuron(gen);
                network->injectCurrent(neuron_id, stim_current(gen) * 0.5);
            }
        }
        
        // Neuromodulation
        if (step % 2000 == 1000) {
            network->releaseNeuromodulator("dopamine", 0.8);
        }
        
        network->step(TIMESTEP);
        
        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        step_times.push_back(step_time);
        
        // Progress reporting
        if (step % 2000 == 0) {
            auto current_stats = network->calculateNetworkStats(50.0);
            std::cout << "  Step " << step << ": Firing rate = " 
                      << std::fixed << std::setprecision(1) << current_stats.mean_firing_rate 
                      << " Hz, Active = " << current_stats.active_neurons << std::endl;
            spike_count += current_stats.active_neurons;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate metrics
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_step_time = total_time / SIMULATION_STEPS;
    
    metrics.total_simulation_time_ms = total_time;
    metrics.average_step_time_ms = avg_step_time;
    metrics.total_spikes_generated = spike_count;
    metrics.firing_rate_hz = (double)spike_count / SIMULATION_DURATION * 1000.0;
    
    // Get performance statistics
    auto perf_stats = network->getPerformanceStats();
    metrics.gpu_utilization_percent = (double)perf_stats.gpu_kernel_calls / 
                                     (perf_stats.gpu_kernel_calls + perf_stats.cpu_step_calls) * 100.0;
    metrics.cpu_fallback_count = perf_stats.cpu_step_calls;
    metrics.cuda_overhead_ms = perf_stats.memory_transfer_time;  // Use memory transfer time as overhead metric
    
    // Memory usage
    metrics.gpu_memory_used_mb = network->estimateGPUMemoryUsage() / 1024 / 1024;
    
    // Network analysis
    auto network_stats = network->calculateNetworkStats(100.0);
    metrics.network_synchrony = network_stats.network_synchrony;
    
    // Estimate performance metrics
    size_t neurons = TARGET_NEURONS;
    size_t synapses = network->getNumSynapses();
    double memory_accessed = (neurons * 64 + synapses * 16) * SIMULATION_STEPS / 1024.0 / 1024.0 / 1024.0; // GB
    metrics.memory_bandwidth_gb_s = memory_accessed / (total_time / 1000.0);
    
    metrics.kernel_efficiency_percent = std::min(100.0, metrics.memory_bandwidth_gb_s / 900.0 * 100.0); // Assume 900 GB/s peak
    
    return metrics;
}

/**
 * @brief Test CUDA memory optimization for 1024 neurons
 */
void testMemoryOptimization() {
    std::cout << "\n=== Memory Optimization Test ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available\n";
        return;
    }
    
    auto config = createOptimal1024Config();
    auto network = std::make_shared<NetworkCUDA>(config);
    
    std::cout << "GPU Memory Analysis:\n";
    std::cout << "  Available GPU Memory: " << network->getAvailableGPUMemory() / 1024 / 1024 << " MB\n";
    std::cout << "  Estimated Memory for 1024 neurons: " << network->estimateGPUMemoryUsage() / 1024 / 1024 << " MB\n";
    
    bool can_fit = NetworkCUDAUtils::canFitOnGPU(TARGET_NEURONS, TARGET_NEURONS * 10);
    std::cout << "  Can fit 1024-neuron network: " << (can_fit ? "Yes" : "No") << "\n";
    
    if (can_fit) {
        size_t optimal_batch = NetworkCUDAUtils::calculateOptimalBatchSize(
            network->getAvailableGPUMemory(), TARGET_NEURONS);
        std::cout << "  Optimal batch size: " << optimal_batch << "\n";
        
        auto optimal_cuda_config = NetworkCUDAUtils::getOptimalCUDAConfig(TARGET_NEURONS, TARGET_NEURONS * 10);
        std::cout << "  Recommended configuration:\n" << optimal_cuda_config.toString() << "\n";
    }
}

/**
 * @brief Test scaling from small to 1024 neurons
 */
void testScalingPerformance() {
    std::cout << "\n=== Scaling Performance Test ===\n";
    
    if (!NetworkCUDAUtils::isCUDAAvailable()) {
        std::cout << "CUDA not available\n";
        return;
    }
    
    std::vector<size_t> neuron_counts = {64, 128, 256, 512, 1024};
    
    std::cout << std::setw(10) << "Neurons" 
              << std::setw(15) << "Avg Step (ms)" 
              << std::setw(15) << "Real-time Factor"
              << std::setw(15) << "GPU Memory (MB)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (size_t neuron_count : neuron_counts) {
        auto config = createOptimal1024Config();
        config.max_neurons = neuron_count;
        
        NetworkBuilderCUDA builder;
        auto network = builder
            .setCUDAConfig(config)
            .enableCUDA(true)
            .addNeuronPopulation(NeuronFactory::PYRAMIDAL_CORTICAL, 
                               neuron_count * 0.8, Position3D(0, 0, 0), 200.0)
            .addNeuronPopulation(NeuronFactory::INTERNEURON_BASKET, 
                               neuron_count * 0.2, Position3D(0, 0, 0), 200.0)
            .addRandomConnections(0.1)
            .buildCUDA();
        
        // Run short benchmark
        const int test_steps = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < test_steps; ++step) {
            if (step % 100 == 0) {
                for (size_t i = 0; i < neuron_count / 20; ++i) {
                    network->injectCurrent(i, 10.0);
                }
            }
            network->step(TIMESTEP);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_step_time = total_time / test_steps;
        double real_time_factor = TIMESTEP / avg_step_time;
        size_t gpu_memory = network->estimateGPUMemoryUsage() / 1024 / 1024;
        
        std::cout << std::setw(10) << neuron_count
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_step_time
                  << std::setw(15) << std::setprecision(2) << real_time_factor
                  << std::setw(15) << gpu_memory << std::endl;
    }
}

/**
 * @brief Save detailed results to file
 */
void saveResults(const DetailedPerformanceMetrics& metrics) {
    std::ofstream file("1024_neuron_cuda_results.txt");
    if (file.is_open()) {
        file << "1024-Neuron CUDA Performance Results\n";
        file << "====================================\n\n";
        file << "Test Configuration:\n";
        file << "  Target Neurons: " << TARGET_NEURONS << "\n";
        file << "  Simulation Steps: " << SIMULATION_STEPS << "\n";
        file << "  Timestep: " << TIMESTEP << " ms\n";
        file << "  Total Simulation Time: " << SIMULATION_DURATION << " ms\n\n";
        
        file << "Performance Results:\n";
        file << "  Real Execution Time: " << metrics.total_simulation_time_ms << " ms\n";
        file << "  Average Step Time: " << metrics.average_step_time_ms << " ms\n";
        file << "  Real-time Factor: " << (TIMESTEP / metrics.average_step_time_ms) << "x\n";
        file << "  GPU Utilization: " << metrics.gpu_utilization_percent << "%\n";
        file << "  Memory Bandwidth: " << metrics.memory_bandwidth_gb_s << " GB/s\n";
        file << "  Total Spikes: " << metrics.total_spikes_generated << "\n";
        file << "  Network Firing Rate: " << metrics.firing_rate_hz << " Hz\n";
        file << "  GPU Memory Used: " << metrics.gpu_memory_used_mb << " MB\n";
        
        auto current_time = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(current_time);
        file << "\nGenerated: " << std::ctime(&time_t);
        
        file.close();
        std::cout << "Results saved to 1024_neuron_cuda_results.txt\n";
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "CUDA-Optimized 1024-Neuron Neural Network Performance Test\n";
    std::cout << "=================================================================\n\n";
    
    try {
        // Check CUDA availability
        if (!NetworkCUDAUtils::isCUDAAvailable()) {
            std::cout << "ERROR: CUDA not available on this system.\n";
            std::cout << "This test requires CUDA-capable hardware and drivers.\n";
            return 1;
        }
        
        std::cout << "CUDA Device Info: " << NetworkCUDAUtils::getCUDADeviceInfo(0) << "\n";
        std::cout << "CUDA Devices Available: " << NetworkCUDAUtils::getCUDADeviceCount() << "\n\n";
        
        // Memory optimization test
        testMemoryOptimization();
        
        // Scaling performance test
        testScalingPerformance();
        
        // Create the main 1024-neuron network
        auto network = create1024NeuronNetwork();
        
        std::cout << "\nNetwork Configuration:\n";
        std::cout << "  Total Neurons: " << network->getNumNeurons() << "\n";
        std::cout << "  Total Synapses: " << network->getNumSynapses() << "\n";
        std::cout << "  CUDA Enabled: " << (network->isCUDAEnabled() ? "Yes" : "No") << "\n";
        std::cout << "  GPU Memory Usage: " << network->estimateGPUMemoryUsage() / 1024 / 1024 << " MB\n";
        
        // Run comprehensive benchmark
        auto metrics = runPerformanceBenchmark(network);
        
        // Print results
        metrics.print();
        
        // Performance assessment
        std::cout << "\n=== Performance Assessment ===\n";
        if (metrics.average_step_time_ms < TIMESTEP) {
            std::cout << "✓ REAL-TIME CAPABLE: Network can run faster than real-time!\n";
            std::cout << "  Real-time factor: " << std::fixed << std::setprecision(2) 
                      << (TIMESTEP / metrics.average_step_time_ms) << "x\n";
        } else {
            std::cout << "⚠ SLOWER THAN REAL-TIME: Network cannot maintain real-time performance\n";
            std::cout << "  Real-time factor: " << std::fixed << std::setprecision(2) 
                      << (TIMESTEP / metrics.average_step_time_ms) << "x\n";
        }
        
        if (metrics.gpu_utilization_percent > 80.0) {
            std::cout << "✓ EXCELLENT GPU UTILIZATION: " << std::setprecision(1) 
                      << metrics.gpu_utilization_percent << "%\n";
        } else if (metrics.gpu_utilization_percent > 50.0) {
            std::cout << "◐ MODERATE GPU UTILIZATION: " << std::setprecision(1) 
                      << metrics.gpu_utilization_percent << "%\n";
        } else {
            std::cout << "⚠ LOW GPU UTILIZATION: " << std::setprecision(1) 
                      << metrics.gpu_utilization_percent << "% - Consider CPU fallback\n";
        }
        
        if (metrics.memory_bandwidth_gb_s > 100.0) {
            std::cout << "✓ GOOD MEMORY BANDWIDTH: " << std::setprecision(1) 
                      << metrics.memory_bandwidth_gb_s << " GB/s\n";
        } else {
            std::cout << "⚠ LIMITED MEMORY BANDWIDTH: " << std::setprecision(1) 
                      << metrics.memory_bandwidth_gb_s << " GB/s\n";
        }
        
        // Save results
        saveResults(metrics);
        
        // Recommendations
        std::cout << "\n=== Optimization Recommendations ===\n";
        if (metrics.average_step_time_ms > TIMESTEP * 0.5) {
            std::cout << "• Consider reducing network connectivity or neuron complexity\n";
        }
        if (metrics.gpu_utilization_percent < 70.0) {
            std::cout << "• Increase batch size or enable adaptive processing\n";
        }
        if (metrics.memory_bandwidth_gb_s < 200.0) {
            std::cout << "• Optimize memory access patterns or use memory coalescing\n";
        }
        if (metrics.cpu_fallback_count > SIMULATION_STEPS * 0.1) {
            std::cout << "• Too many CPU fallbacks - check GPU memory limits\n";
        }
        
        std::cout << "\n=== Test Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

/*
COMPILATION INSTRUCTIONS FOR 1024-NEURON CUDA TEST:
==================================================

Prerequisites:
- NVIDIA GPU with Compute Capability 3.5+
- CUDA Toolkit 11.0 or later
- 4GB+ GPU memory (recommended for 1024 neurons)

Compile with maximum optimization:
nvcc -std=c++14 -O3 -use_fast_math -arch=sm_75 \
     -o main_1024_cuda main_1024_CUDA.cpp Network_CUDA.cpp NetworkCUDA.cu \
     Network.cpp Neuron.cpp -lcudart -lcublas -lcurand

For debugging:
nvcc -std=c++14 -g -G -lineinfo \
     -o main_1024_cuda_debug main_1024_CUDA.cpp Network_CUDA.cpp NetworkCUDA.cu \
     Network.cpp Neuron.cpp -lcudart -lcublas -lcurand

Expected Performance Targets:
- Real-time factor: >1.0x for 1024 neurons
- GPU utilization: >80%
- Memory bandwidth: >200 GB/s
- Average step time: <0.01ms

Memory Requirements:
- ~80MB GPU memory for 1024 neurons
- ~2GB system memory
- Peak memory usage during simulation: ~150MB

Performance Notes:
- Optimal for networks with 500-2000 neurons
- CUDA kernels optimized for high occupancy
- Memory access patterns optimized for coalescing
- Supports concurrent kernel execution
- Adaptive batch sizing for optimal performance

Troubleshooting:
- If GPU memory insufficient: Reduce max_neurons or batch_size
- If poor performance: Check CUDA device compatibility
- If frequent CPU fallbacks: Increase GPU memory pool size
*/
