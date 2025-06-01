/**
 * @file Network_CUDA.cpp
 * @brief Implementation of CUDA-enhanced neural network
 */

#include "Network_CUDA.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#endif

// ============================================================================
// NetworkCUDA Implementation
// ============================================================================

NetworkCUDA::NetworkCUDA(const NetworkConfigCUDA& config) 
    : Network(config), cuda_config_(config), cuda_available_(false), cuda_enabled_(false)
#ifdef USE_CUDA
    , cuda_processor_(nullptr), cuda_memory_(nullptr), total_gpu_time_(0.0f), 
      total_cpu_time_(0.0f), gpu_kernel_calls_(0), cpu_step_calls_(0),
      data_on_gpu_(false), gpu_initialized_(false), main_stream_(0), 
      memory_stream_(0), use_gpu_this_step_(false), gpu_switch_threshold_(100)
#endif
{
    cuda_available_ = checkCUDAAvailability();
    
    if (cuda_available_ && config.enable_cuda) {
        if (initializeCUDA()) {
            cuda_enabled_ = true;
            std::cout << "CUDA acceleration enabled successfully" << std::endl;
        } else {
            std::cout << "CUDA initialization failed, falling back to CPU" << std::endl;
            cuda_enabled_ = false;
        }
    }
}

NetworkCUDA::~NetworkCUDA() {
#ifdef USE_CUDA
    cleanupCUDA();
#endif
}

bool NetworkCUDA::checkCUDAAvailability() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return false;
    }
    
    // Check if we can create a context on the default device
    cudaError_t context_error = cudaSetDevice(cuda_config_.cuda_device_id);
    if (context_error != cudaSuccess) {
        return false;
    }
    
    return true;
#else
    return false;
#endif
}

bool NetworkCUDA::initializeCUDA() {
#ifdef USE_CUDA
    if (gpu_initialized_) return true;
    
    try {
        // Set device
        CUDA_CHECK(cudaSetDevice(cuda_config_.cuda_device_id));
        
        // Create streams
        if (cuda_config_.async_memory_transfer) {
            CUDA_CHECK(cudaStreamCreate(&main_stream_));
            CUDA_CHECK(cudaStreamCreate(&memory_stream_));
        }
        
        // Initialize CUDA processor
        cuda_processor_ = std::make_unique<CUDANetworkProcessor>();
        if (!cuda_processor_->initialize(cuda_config_.max_neurons, 10000)) { // Estimate max synapses
            return false;
        }
        
        // Initialize memory manager
        cuda_memory_ = std::make_unique<CUDAMemoryManager>();
        if (!cuda_memory_->initialize(cuda_config_.max_neurons, 10000, 1000)) {
            return false;
        }
        
        // Set up GPU data structures
        gpu_neurons_.resize(cuda_config_.max_neurons);
        gpu_synapses_.resize(10000);
        
        gpu_initialized_ = true;
        data_on_gpu_ = false;
        
        std::cout << "CUDA initialized successfully on device " << cuda_config_.cuda_device_id << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization error: " << e.what() << std::endl;
        cleanupCUDA();
        return false;
    }
#else
    return false;
#endif
}

#ifdef USE_CUDA
void NetworkCUDA::cleanupCUDA() {
    if (main_stream_) {
        cudaStreamDestroy(main_stream_);
        main_stream_ = 0;
    }
    
    if (memory_stream_) {
        cudaStreamDestroy(memory_stream_);
        memory_stream_ = 0;
    }
    
    cuda_processor_.reset();
    cuda_memory_.reset();
    
    gpu_neurons_.clear();
    gpu_synapses_.clear();
    
    gpu_initialized_ = false;
    data_on_gpu_ = false;
}

GPUNeuronState NetworkCUDA::convertNeuronToGPU(size_t neuron_id) {
    GPUNeuronState gpu_neuron = {};
    
    if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) {
        gpu_neuron.active = 0;
        return gpu_neuron;
    }
    
    auto& cpu_neuron = neurons_[neuron_id];
    gpu_neuron.active = 1;
    
    // Convert position
    if (neuron_id < neuron_positions_.size()) {
        gpu_neuron.x = static_cast<float>(neuron_positions_[neuron_id].x);
        gpu_neuron.y = static_cast<float>(neuron_positions_[neuron_id].y);
        gpu_neuron.z = static_cast<float>(neuron_positions_[neuron_id].z);
    }
    
    // Convert spike information
    gpu_neuron.last_spike_time = static_cast<float>(cpu_neuron->getLastSpikeTime());
    gpu_neuron.spike_threshold = -20.0f; // Default threshold
    
    // Convert compartment data (simplified mapping)
    gpu_neuron.compartment_count = 4; // Standard morphology
    
    // Initialize default values for compartments
    for (int comp = 0; comp < MAX_COMPARTMENTS; ++comp) {
        gpu_neuron.voltages[comp] = -65.0f; // Resting potential
        gpu_neuron.capacitances[comp] = 1.0f; // Default capacitance
        gpu_neuron.areas[comp] = 1e-8f; // Default area
        
        // Initialize ion channels
        for (int ch = 0; ch < MAX_ION_CHANNELS; ++ch) {
            gpu_neuron.channel_states[comp][ch][0] = 0.05f; // m
            gpu_neuron.channel_states[comp][ch][1] = 0.6f;  // h  
            gpu_neuron.channel_states[comp][ch][2] = 0.32f; // n
            
            // Set conductances based on compartment type
            if (comp == 1) { // Soma
                if (ch == 0) { // Sodium
                    gpu_neuron.channel_conductances[comp][ch] = 50.0f;
                    gpu_neuron.channel_reversals[comp][ch] = 50.0f;
                } else if (ch == 1) { // Potassium
                    gpu_neuron.channel_conductances[comp][ch] = 20.0f;
                    gpu_neuron.channel_reversals[comp][ch] = -77.0f;
                }
            }
        }
        
        // Initialize synaptic receptors
        for (int rec = 0; rec < MAX_SYNAPTIC_RECEPTORS; ++rec) {
            gpu_neuron.receptor_conductances[comp][rec] = 0.0f;
            gpu_neuron.receptor_reversals[comp][rec] = 0.0f;
            gpu_neuron.receptor_tau_rise[comp][rec] = 0.2f;
            gpu_neuron.receptor_tau_decay[comp][rec] = 2.0f;
        }
    }
    
    return gpu_neuron;
}

GPUSynapse NetworkCUDA::convertSynapseToGPU(size_t synapse_id) {
    GPUSynapse gpu_synapse = {};
    
    if (synapse_id >= synapses_.size() || !synapses_[synapse_id]) {
        gpu_synapse.active = 0;
        return gpu_synapse;
    }
    
    auto& cpu_synapse = synapses_[synapse_id];
    gpu_synapse.active = 1;
    
    gpu_synapse.pre_neuron_id = static_cast<int>(cpu_synapse->pre_neuron_id);
    gpu_synapse.post_neuron_id = static_cast<int>(cpu_synapse->post_neuron_id);
    gpu_synapse.post_compartment = 1; // Default to soma
    gpu_synapse.receptor_index = static_cast<int>(cpu_synapse->receptor_index);
    
    gpu_synapse.weight = static_cast<float>(cpu_synapse->weight);
    gpu_synapse.base_weight = static_cast<float>(cpu_synapse->base_weight);
    gpu_synapse.delay = static_cast<float>(cpu_synapse->axonal_delay);
    
    gpu_synapse.last_pre_spike = static_cast<float>(cpu_synapse->last_pre_spike);
    gpu_synapse.last_post_spike = static_cast<float>(cpu_synapse->last_post_spike);
    gpu_synapse.eligibility_trace = static_cast<float>(cpu_synapse->eligibility_trace);
    gpu_synapse.activity_metric = static_cast<float>(cpu_synapse->activity_metric);
    gpu_synapse.last_potentiation = static_cast<float>(cpu_synapse->last_potentiation);
    
    return gpu_synapse;
}

void NetworkCUDA::convertNeuronsToGPU() {
    for (size_t i = 0; i < std::min(neurons_.size(), gpu_neurons_.size()); ++i) {
        gpu_neurons_[i] = convertNeuronToGPU(i);
    }
}

void NetworkCUDA::convertSynapsesToGPU() {
    for (size_t i = 0; i < std::min(synapses_.size(), gpu_synapses_.size()); ++i) {
        gpu_synapses_[i] = convertSynapseToGPU(i);
    }
}

void NetworkCUDA::convertDataFromGPU() {
    // Convert neuron data back to CPU format
    for (size_t i = 0; i < std::min(gpu_neurons_.size(), neurons_.size()); ++i) {
        if (gpu_neurons_[i].active && neurons_[i]) {
            // Note: We can't directly update private neuron members
            // In a real implementation, you'd need accessor methods in the Neuron class
            // For now, we'll skip direct member updates
            
            // The GPU simulation results are maintained in the GPU state
            // and would need proper accessor methods to update CPU state
        }
    }
    
    // Convert synapse data back
    for (size_t i = 0; i < std::min(gpu_synapses_.size(), synapses_.size()); ++i) {
        if (gpu_synapses_[i].active && synapses_[i]) {
            synapses_[i]->weight = gpu_synapses_[i].weight;
            synapses_[i]->activity_metric = gpu_synapses_[i].activity_metric;
            synapses_[i]->last_pre_spike = gpu_synapses_[i].last_pre_spike;
            synapses_[i]->last_post_spike = gpu_synapses_[i].last_post_spike;
            synapses_[i]->eligibility_trace = gpu_synapses_[i].eligibility_trace;
            synapses_[i]->last_potentiation = gpu_synapses_[i].last_potentiation;
        }
    }
}

bool NetworkCUDA::shouldUseGPU() const {
#ifdef USE_CUDA
    if (!cuda_enabled_ || !gpu_initialized_) return false;
    
    if (cuda_config_.adaptive_processing) {
        // Use GPU if we have enough neurons to make it worthwhile
        return getActiveNeuronCount() >= gpu_switch_threshold_;
    }
    
    return true;
#else
    return false;
#endif
}

void NetworkCUDA::handleCUDAError(const std::string& operation) {
    std::cerr << "CUDA error in operation: " << operation << std::endl;
    // For now, just fall back to CPU
    cuda_enabled_ = false;
}
#endif

void NetworkCUDA::fallbackToCPU() {
    std::cout << "Falling back to CPU processing" << std::endl;
    cuda_enabled_ = false;
}

void NetworkCUDA::uploadToGPU() {
#ifdef USE_CUDA
    if (!cuda_enabled_ || !gpu_initialized_ || data_on_gpu_) return;
    
    try {
        convertNeuronsToGPU();
        convertSynapsesToGPU();
        
        cuda_processor_->uploadNetworkData(gpu_neurons_, gpu_synapses_);
        
        data_on_gpu_ = true;
    } catch (const std::exception& e) {
        handleCUDAError("uploadToGPU");
    }
#endif
}

void NetworkCUDA::downloadFromGPU() {
#ifdef USE_CUDA
    if (!cuda_enabled_ || !gpu_initialized_ || !data_on_gpu_) return;
    
    try {
        cuda_processor_->downloadNetworkData(gpu_neurons_, gpu_synapses_);
        convertDataFromGPU();
        
        data_on_gpu_ = false;
    } catch (const std::exception& e) {
        handleCUDAError("downloadFromGPU");
    }
#endif
}

void NetworkCUDA::synchronizeGPUData() {
#ifdef USE_CUDA
    if (!cuda_enabled_ || !gpu_initialized_) return;
    
    if (data_on_gpu_) {
        downloadFromGPU();
    }
    uploadToGPU();
#endif
}

void NetworkCUDA::setGPUEnabled(bool enabled) {
    if (enabled && !cuda_available_) {
        std::cout << "Cannot enable CUDA: not available on this system" << std::endl;
        return;
    }
    
    if (enabled && !cuda_enabled_) {
        if (initializeCUDA()) {
            cuda_enabled_ = true;
            std::cout << "CUDA enabled" << std::endl;
        }
    } else if (!enabled && cuda_enabled_) {
#ifdef USE_CUDA
        if (data_on_gpu_) {
            downloadFromGPU();
        }
#endif
        cuda_enabled_ = false;
        std::cout << "CUDA disabled" << std::endl;
    }
}

std::string NetworkCUDA::getCUDADeviceInfo() const {
#ifdef USE_CUDA
    if (!cuda_available_) return "CUDA not available";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_config_.cuda_device_id);
    
    return std::string("Device: ") + prop.name + 
           ", Compute Capability: " + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
           ", Global Memory: " + std::to_string(prop.totalGlobalMem / 1024 / 1024) + " MB" +
           ", Multiprocessors: " + std::to_string(prop.multiProcessorCount);
#else
    return "CUDA support not compiled";
#endif
}

bool NetworkCUDA::setCUDADevice(int device_id) {
#ifdef USE_CUDA
    if (!cuda_available_) return false;
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_id < 0 || device_id >= device_count) {
        return false;
    }
    
    cuda_config_.cuda_device_id = device_id;
    
    if (cuda_enabled_) {
        // Re-initialize with new device
        cleanupCUDA();
        return initializeCUDA();
    }
    
    return true;
#else
    (void)device_id;
    return false;
#endif
}

NetworkCUDA::CUDAPerformanceStats NetworkCUDA::getPerformanceStats() const {
    CUDAPerformanceStats stats = {};
    
#ifdef USE_CUDA
    stats.total_gpu_time = total_gpu_time_;
    stats.total_cpu_time = total_cpu_time_;
    stats.gpu_kernel_calls = gpu_kernel_calls_;
    stats.cpu_step_calls = cpu_step_calls_;
    
    if (total_cpu_time_ > 0) {
        stats.gpu_speedup = total_cpu_time_ / total_gpu_time_;
    }
    
    if (cuda_processor_) {
        stats.total_gpu_time = cuda_processor_->getGPUTime();
        stats.gpu_kernel_calls = cuda_processor_->getKernelCalls();
    }
#endif
    
    return stats;
}

void NetworkCUDA::resetPerformanceCounters() {
#ifdef USE_CUDA
    total_gpu_time_ = 0.0f;
    total_cpu_time_ = 0.0f;
    gpu_kernel_calls_ = 0;
    cpu_step_calls_ = 0;
    
    if (cuda_processor_) {
        cuda_processor_->resetPerformanceCounters();
    }
#endif
}

// ============================================================================
// Overridden Network Methods with CUDA Acceleration
// ============================================================================

void NetworkCUDA::step(double dt) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
#ifdef USE_CUDA
    use_gpu_this_step_ = shouldUseGPU();
    
    if (use_gpu_this_step_) {
        // GPU processing path
        if (!data_on_gpu_) {
            uploadToGPU();
        }
        
        cuda_processor_->setCurrentSizes(getActiveNeuronCount(), getSynapseCount());
        cuda_processor_->updateNetwork(static_cast<float>(dt), static_cast<float>(getCurrentTime()));
        
        if (getConfig().enable_stdp) {
            cuda_processor_->updatePlasticity(
                static_cast<float>(getConfig().stdp_learning_rate),
                static_cast<float>(getConfig().stdp_tau_pre),
                static_cast<float>(getConfig().stdp_tau_post),
                static_cast<float>(getCurrentTime())
            );
        }
        
        gpu_kernel_calls_++;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_gpu_time_ += duration.count() / 1000.0f; // Convert to milliseconds
    } else {
#endif
        // CPU processing path (fallback or small networks)
#ifdef USE_CUDA
        if (data_on_gpu_) {
            downloadFromGPU();
        }
#endif
        
        Network::step(dt); // Call parent implementation
        
#ifdef USE_CUDA
        cpu_step_calls_++;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_cpu_time_ += duration.count() / 1000.0f;
    }
#endif
    
    // Update time through the protected method
    setCurrentTime(getCurrentTime() + dt);
}

void NetworkCUDA::run(double duration, double dt_sim) {
#ifdef USE_CUDA
    // For long runs, keep data on GPU if using CUDA
    bool keep_on_gpu = cuda_enabled_ && shouldUseGPU();
    
    if (keep_on_gpu && !data_on_gpu_) {
        uploadToGPU();
    }
#endif
    
    Network::run(duration, dt_sim);
    
#ifdef USE_CUDA
    // Download final state if needed
    if (keep_on_gpu && data_on_gpu_) {
        downloadFromGPU();
    }
#endif
}

void NetworkCUDA::updatePlasticity() {
#ifdef USE_CUDA
    if (use_gpu_this_step_ && cuda_enabled_) {
        // STDP is handled in the main GPU step
        return;
    }
#endif
    
    Network::updatePlasticity();
}

void NetworkCUDA::performSynaptogenesis() {
    // For now, fall back to CPU implementation
    // GPU implementation would require more complex connectivity management
#ifdef USE_CUDA
    if (data_on_gpu_) {
        downloadFromGPU();
    }
#endif
    
    Network::performSynaptogenesis();
    
    // Force GPU data refresh on next step
#ifdef USE_CUDA
    data_on_gpu_ = false;
#endif
}

void NetworkCUDA::performNeurogenesis() {
    // For now, fall back to CPU implementation
#ifdef USE_CUDA
    if (data_on_gpu_) {
        downloadFromGPU();
    }
#endif
    
    Network::performNeurogenesis();
    
    // Force GPU data refresh on next step
#ifdef USE_CUDA
    data_on_gpu_ = false;
#endif
}

void NetworkCUDA::performPruning() {
    // For now, fall back to CPU implementation
#ifdef USE_CUDA
    if (data_on_gpu_) {
        downloadFromGPU();
    }
#endif
    
    Network::performPruning();
    
    // Force GPU data refresh on next step
#ifdef USE_CUDA
    data_on_gpu_ = false;
#endif
}

void NetworkCUDA::injectCurrent(size_t neuron_id, double current) {
#ifdef USE_CUDA
    if (use_gpu_this_step_ && cuda_enabled_ && data_on_gpu_) {
        cuda_processor_->injectCurrent(static_cast<int>(neuron_id), static_cast<float>(current));
        return;
    }
#endif
    
    Network::injectCurrent(neuron_id, current);
}

void NetworkCUDA::releaseNeuromodulator(const std::string& type, double amount) {
#ifdef USE_CUDA
    if (use_gpu_this_step_ && cuda_enabled_ && data_on_gpu_ && type == "dopamine") {
        cuda_processor_->releaseNeuromodulator(static_cast<float>(amount), 
                                              static_cast<float>(config_.modulation_strength));
        return;
    }
#endif
    
    Network::releaseNeuromodulator(type, amount);
}

Network::NetworkStats NetworkCUDA::calculateNetworkStats(double time_window) const {
#ifdef USE_CUDA
    if (cuda_enabled_ && data_on_gpu_) {
        // GPU-accelerated statistics calculation
        float stats_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Note: This would require a const-correct version of calculateStatistics
        // For now, fall back to CPU implementation
        return Network::calculateNetworkStats(time_window);
    }
#endif
    
    return Network::calculateNetworkStats(time_window);
}

std::vector<double> NetworkCUDA::getRegionalActivity(size_t x_bins, size_t y_bins, double time_window) const {
    // For now, use CPU implementation
    // GPU version would require implementing regional binning kernels
    return Network::getRegionalActivity(x_bins, y_bins, time_window);
}

size_t NetworkCUDA::estimateGPUMemoryUsage() const {
#ifdef USE_CUDA
    size_t neuron_memory = getActiveNeuronCount() * sizeof(GPUNeuronState);
    size_t synapse_memory = getSynapseCount() * sizeof(GPUSynapse);
    size_t overhead = 100 * 1024 * 1024; // 100MB overhead
    
    return neuron_memory + synapse_memory + overhead;
#else
    return 0;
#endif
}

size_t NetworkCUDA::getAvailableGPUMemory() const {
#ifdef USE_CUDA
    if (!cuda_available_) return 0;
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
#else
    return 0;
#endif
}

void NetworkCUDA::optimizeMemoryLayout() {
    // Implement memory layout optimization
    // For now, this is a placeholder
}

void NetworkCUDA::updateCUDAConfig(const NetworkConfigCUDA& config) {
    cuda_config_ = config;
    
    if (config.enable_cuda && !cuda_enabled_) {
        setGPUEnabled(true);
    } else if (!config.enable_cuda && cuda_enabled_) {
        setGPUEnabled(false);
    }
}

// ============================================================================
// NetworkBuilderCUDA Implementation
// ============================================================================

NetworkBuilderCUDA::NetworkBuilderCUDA() : connectionProbability_(0.0) {
    cuda_config_ = NetworkConfigCUDA();
}

NetworkBuilderCUDA& NetworkBuilderCUDA::addNeuronPopulation(NeuronFactory::NeuronType type,
                                                           size_t count,
                                                           const Position3D& position,
                                                           double radius) {
    NeuronPopulation pop;
    pop.type = type;
    pop.count = count;
    pop.position = position;
    pop.radius = radius;
    neuronPopulations_.push_back(pop);
    return *this;
}

NetworkBuilderCUDA& NetworkBuilderCUDA::addRandomConnections(double probability) {
    connectionProbability_ = std::max(0.0, std::min(1.0, probability));
    return *this;
}

Position3D NetworkBuilderCUDA::randomizePosition(const Position3D& center, double radius) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    Position3D pos;
    do {
        pos.x = center.x + radius * dis(gen);
        pos.y = center.y + radius * dis(gen);
        pos.z = center.z + radius * dis(gen);
    } while (pos.distanceTo(center) > radius);
    
    return pos;
}

NetworkBuilderCUDA& NetworkBuilderCUDA::setCUDAConfig(const NetworkConfigCUDA& config) {
    cuda_config_ = config;
    return *this;
}

NetworkBuilderCUDA& NetworkBuilderCUDA::enableCUDA(bool enable) {
    cuda_config_.enable_cuda = enable;
    return *this;
}

NetworkBuilderCUDA& NetworkBuilderCUDA::setCUDADevice(int device_id) {
    cuda_config_.cuda_device_id = device_id;
    return *this;
}

NetworkBuilderCUDA& NetworkBuilderCUDA::setGPUMemoryLimit(size_t limit_bytes) {
    cuda_config_.gpu_memory_limit = limit_bytes;
    return *this;
}

std::shared_ptr<NetworkCUDA> NetworkBuilderCUDA::buildCUDA() {
    auto network = std::make_shared<NetworkCUDA>(cuda_config_);
    
    // Apply neuron populations from base class
    for (const auto& pop : neuronPopulations_) {
        for (size_t i = 0; i < pop.count; ++i) {
            Position3D neuron_pos = randomizePosition(pop.position, pop.radius);
            std::string neuron_id = "neuron_" + std::to_string(i) + "_pop_" + std::to_string(&pop - &neuronPopulations_[0]);
            
            auto neuron = NeuronFactory::createNeuron(pop.type, neuron_id, neuron_pos);
            network->addNeuron(neuron, neuron_pos);
        }
    }
    
    // Apply random connections if specified
    if (connectionProbability_ > 0.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        size_t total_neurons = network->getNumNeurons();
        
        for (size_t i = 0; i < total_neurons; ++i) {
            for (size_t j = 0; j < total_neurons; ++j) {
                if (i != j && dis(gen) < connectionProbability_) {
                    double weight = 0.1 + 0.4 * dis(gen); // Random weight 0.1-0.5
                    network->createSynapse(i, j, "dendrite", 0, weight);
                }
            }
        }
    }
    
    return network;
}

std::shared_ptr<NetworkCUDA> NetworkBuilderCUDA::buildAuto() {
    // Auto-detect best configuration
    cuda_config_.enable_cuda = NetworkCUDAUtils::isCUDAAvailable();
    if (cuda_config_.enable_cuda) {
        cuda_config_ = NetworkCUDAUtils::recommendCUDASettings();
    }
    
    return buildCUDA();
}

// ============================================================================
// NetworkCUDAUtils Implementation
// ============================================================================

namespace NetworkCUDAUtils {

bool isCUDAAvailable() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

int getCUDADeviceCount() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

std::string getCUDADeviceInfo(int device_id) {
#ifdef USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return std::string(prop.name) + " (CC " + 
               std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
    }
#endif
    (void)device_id;
    return "No device info available";
}

NetworkConfigCUDA getOptimalCUDAConfig(size_t num_neurons, size_t num_synapses) {
    NetworkConfigCUDA config;
    config.enable_cuda = isCUDAAvailable();
    if (!config.enable_cuda) return config;
    if (num_neurons < 50) {
        config.enable_cuda = false;
    } else if (num_neurons < 500) {
        config.adaptive_processing = true;
        config.gpu_load_threshold = 100;
    } else {
        config.adaptive_processing = false;
    }
    config.max_neurons = std::max(num_neurons * 2, static_cast<size_t>(1000));
    return config;
}

NetworkConfigCUDA recommendCUDASettings() {
    NetworkConfigCUDA config;
    if (!isCUDAAvailable()) {
        config.enable_cuda = false;
        return config;
    }
#ifdef USE_CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    config.enable_cuda = true;
    config.cuda_device_id = 0;
    config.adaptive_processing = true;
    if (prop.totalGlobalMem > 8ULL * 1024 * 1024 * 1024) {
        config.max_neurons = 10000;
        config.gpu_load_threshold = 50;
    } else if (prop.totalGlobalMem > 4ULL * 1024 * 1024 * 1024) {
        config.max_neurons = 5000;
        config.gpu_load_threshold = 100;
    } else {
        config.max_neurons = 2000;
        config.gpu_load_threshold = 200;
    }
    config.use_pinned_memory = true;
    config.async_memory_transfer = true;
#endif
    return config;
}

bool canFitOnGPU(size_t num_neurons, size_t num_synapses) {
#ifdef USE_CUDA
    if (!isCUDAAvailable()) return false;
    size_t required_memory = num_neurons * sizeof(GPUNeuronState) + 
                           num_synapses * sizeof(GPUSynapse);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return required_memory < (free_mem * 0.8);
#else
    (void)num_neurons; (void)num_synapses;
    return false;
#endif
}

size_t calculateOptimalBatchSize(size_t available_memory, size_t num_neurons) {
#ifdef USE_CUDA
    size_t memory_per_neuron = sizeof(GPUNeuronState) + 100;
    return std::min(num_neurons, available_memory / memory_per_neuron);
#else
    (void)available_memory; (void)num_neurons;
    return 0;
#endif
}

std::string getLastCUDAError() {
#ifdef USE_CUDA
    cudaError_t error = cudaGetLastError();
    return std::string(cudaGetErrorString(error));
#else
    return "CUDA not available";
#endif
}

void clearCUDAErrors() {
#ifdef USE_CUDA
    cudaGetLastError();
#endif
}

} // namespace NetworkCUDAUtils

// ============================================================================
// Factory Functions
// ============================================================================

std::shared_ptr<NetworkType> createOptimalNetwork(const NetworkConfigType& config) {
#ifdef USE_CUDA
    // Try to create CUDA network if available
    if (NetworkCUDAUtils::isCUDAAvailable()) {
        NetworkConfigCUDA cuda_config;
        
        // Copy base config settings
        cuda_config.dt = config.dt;
        cuda_config.axonal_speed = config.axonal_speed;
        cuda_config.network_width = config.network_width;
        cuda_config.network_height = config.network_height;
        cuda_config.network_depth = config.network_depth;
        cuda_config.max_connection_distance = config.max_connection_distance;
        cuda_config.connection_probability_base = config.connection_probability_base;
        cuda_config.distance_decay_constant = config.distance_decay_constant;
        cuda_config.max_neurons = config.max_neurons;
        cuda_config.enable_neurogenesis = config.enable_neurogenesis;
        cuda_config.enable_pruning = config.enable_pruning;
        cuda_config.enable_stdp = config.enable_stdp;
        cuda_config.enable_neuromodulation = config.enable_neuromodulation;
        
        // Enable CUDA
        cuda_config.enable_cuda = true;
        
        return std::make_shared<NetworkCUDA>(cuda_config);
    }
#endif
    
    // Fall back to CPU-only network
    return std::make_shared<Network>(config);
}

std::shared_ptr<NetworkType> createNetwork(const NetworkConfigType& config, bool prefer_cuda) {
#ifdef USE_CUDA
    if (prefer_cuda && NetworkCUDAUtils::isCUDAAvailable()) {
        // Convert to CUDA config if needed
        const NetworkConfigCUDA* cuda_config = dynamic_cast<const NetworkConfigCUDA*>(&config);
        if (cuda_config) {
            NetworkConfigCUDA final_config = *cuda_config;
            final_config.enable_cuda = true;
            return std::make_shared<NetworkCUDA>(final_config);
        } else {
            // Create CUDA config from base config
            NetworkConfigCUDA cuda_config;
            cuda_config.dt = config.dt;
            cuda_config.max_neurons = config.max_neurons;
            cuda_config.enable_cuda = true;
            // Copy other relevant fields...
            return std::make_shared<NetworkCUDA>(cuda_config);
        }
    }
#else
    (void)prefer_cuda; // Suppress unused parameter warning
#endif
    
    return std::make_shared<Network>(config);
}

void NetworkCUDA::step(float dt, float threshold) {
#ifdef USE_CUDA
    if (cuda_enabled_) {
        // Launch CUDA kernels for neuron updates and synapse processing
        launchUpdateNeuronVoltages(gpu_neurons_.data(), nullptr, dt, 0.0f, gpu_neurons_.size());
        launchSynapseInputKernel(gpu_synapses_.data(), gpu_neurons_.data(), gpu_synapses_.size());
        cudaDeviceSynchronize();
        return;
    }
#endif
    // Fallback to CPU implementation if CUDA is not enabled
    Network::step(dt, threshold);
}

int NetworkCUDA::getDeviceSpikeCount() const {
#ifdef USE_CUDA
    if (cuda_enabled_) {
        int spike_count = 0;
        cudaMemcpyFromSymbol(&spike_count, d_spike_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
        return spike_count;
    }
#endif
    return 0; // Fallback if CUDA is not enabled
}