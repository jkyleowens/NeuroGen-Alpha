#include "NetworkCUDA.cuh"

// Include all necessary headers
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/cuda/CudaUtils.h>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/NeuronInitialization.cuh>
#include <NeuroGen/cuda/SynapseInputKernel.cuh>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/cuda/HebbianLearningKernel.cuh>
#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>
#include <NeuroGen/cuda/StructuralPlasticityKernels.cuh>
#include <NeuroGen/cuda/RewardModulationKernel.cuh>
#include <NeuroGen/cuda/RandomStateInit.cuh>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/cuda/CorticalColumn.h>
#include <random>
#include <cmath>
#include <algorithm>


// ====================================================================================
// CONSTRUCTOR / DESTRUCTOR
// ====================================================================================

NetworkCUDA::NetworkCUDA(const NetworkConfig& config) :
    config(config),
    d_neurons(nullptr),
    d_synapses(nullptr),
    d_calcium_levels(nullptr),
    d_neuron_spike_counts(nullptr),
    d_random_states(nullptr),
    d_cortical_columns(nullptr),
    d_input_currents(nullptr),
    current_time_ms(0.0f),
    network_initialized(false),
    plasticity_enabled(true),
    current_learning_rate(0.001f)
{
    std::cout << "Initializing NetworkCUDA..." << std::endl;
    
    try {
        // Finalize the configuration to compute derived values
        const_cast<NetworkConfig&>(this->config).finalizeConfig();
        
        std::cout << "Network config finalized:" << std::endl;
        std::cout << "  Neurons: " << getNumNeurons() << std::endl;
        std::cout << "  Synapses: " << getNumSynapses() << std::endl;
        
        validateConfig();
        allocateDeviceMemory();
        initializeDeviceArrays();
        initializeNetwork();
        initializeColumns();
        
        network_initialized = true;
        g_stats.reset();
        
        std::cout << "NetworkCUDA initialized successfully." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during NetworkCUDA initialization: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

NetworkCUDA::~NetworkCUDA() {
    std::cout << "Cleaning up NetworkCUDA..." << std::endl;
    cleanup();
    std::cout << "NetworkCUDA cleanup finished." << std::endl;
}

// ====================================================================================
// PUBLIC METHODS
// ====================================================================================

void NetworkCUDA::update(float dt_ms, const std::vector<float>& input_currents, float reward) {
    if (!network_initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED, 
                             "Network must be initialized before calling update()");
    }
    
    current_time_ms += dt_ms;
    g_stats.total_simulation_time += dt_ms;
    g_stats.current_reward = reward;

    // Kernel launch configurations
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    int num_synapses = static_cast<int>(config.totalSynapses);

    dim3 neuron_blocks, neuron_threads;
    calculateGridBlockSize(num_neurons, neuron_blocks, neuron_threads);

    dim3 synapse_blocks, synapse_threads;
    calculateGridBlockSize(num_synapses, synapse_blocks, synapse_threads);

    try {
        std::cout << "[DEBUG] Starting network update - num_neurons: " << num_neurons 
                  << ", num_synapses: " << num_synapses 
                  << ", input_size: " << input_currents.size() << std::endl;

        // 1. Apply external input currents
        if (!input_currents.empty()) {
            std::cout << "[DEBUG] Applying input currents..." << std::endl;
            
            // Copy input data from host to device
            int input_size = static_cast<int>(input_currents.size());
            std::cout << "[DEBUG] Copying " << input_size << " input values to device..." << std::endl;
            
            CUDA_CHECK_ERROR(cudaMemcpy(d_input_currents, input_currents.data(), 
                                       input_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Call kernel with device memory
            applyInputCurrentsWrapper(d_neurons, d_input_currents, input_size, num_neurons);
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "[ERROR] Input currents kernel failed: " << cudaGetErrorString(err) << std::endl;
                throw NetworkException(NetworkError::CUDA_ERROR, 
                                     "Input currents kernel error: " + std::string(cudaGetErrorString(err)));
            }
        }

        // 2. Process synaptic inputs  
        std::cout << "[DEBUG] Processing synaptic inputs..." << std::endl;
        processSynapticInputsWrapper(d_neurons, d_synapses, num_synapses, num_neurons);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Synaptic inputs kernel failed: " << cudaGetErrorString(err) << std::endl;
            throw NetworkException(NetworkError::CUDA_ERROR, 
                                 "Synaptic inputs kernel error: " + std::string(cudaGetErrorString(err)));
        }

        // 3. Update neuron states (voltage, recovery, etc.)
        std::cout << "[DEBUG] Updating neuron states..." << std::endl;
        updateNeuronsWrapper(dt_ms);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Neuron update kernel failed: " << cudaGetErrorString(err) << std::endl;
            throw NetworkException(NetworkError::CUDA_ERROR, 
                                 "Neuron update kernel error: " + std::string(cudaGetErrorString(err)));
        }

        // 4. Process neuron spiking
        std::cout << "[DEBUG] Processing neuron spiking..." << std::endl;
        processSpikingWrapper();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Spike processing kernel failed: " << cudaGetErrorString(err) << std::endl;
            throw NetworkException(NetworkError::CUDA_ERROR, 
                                 "Spike processing kernel error: " + std::string(cudaGetErrorString(err)));
        }

        // 5. Update eligibility traces (if plasticity enabled)
        if (plasticity_enabled) {
            updateEligibilityTracesWrapper(synapse_blocks, synapse_threads, 
                                         d_synapses, d_neurons, dt_ms, num_synapses);
            CUDA_KERNEL_CHECK();
        }

        // 6. Apply reward modulation (if plasticity enabled and reward provided)
        if (plasticity_enabled && reward != 0.0f) {
            applyPlasticityWrapper(reward);
        }

        // 7. Apply Hebbian learning (if plasticity enabled)
        if (plasticity_enabled) {
            applyHebbianLearningWrapper(synapse_blocks, synapse_threads, 
                                      d_synapses, d_neurons, num_synapses);
            CUDA_KERNEL_CHECK();
        }

        // 8. Periodic homeostatic scaling
        if (static_cast<int>(current_time_ms / dt_ms) % 1000 == 0) {
            applyHomeostaticScalingWrapper(synapse_blocks, synapse_threads, 
                                         d_synapses, num_synapses);
            CUDA_KERNEL_CHECK();
        }

        // 9. Update network statistics
        updateNetworkStatistics();

        // 10. Final synchronization
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    } catch (const std::exception& e) {
        std::cerr << "Error during network update: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> NetworkCUDA::getOutput() const {
    if (!network_initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED,
                             "Network not initialized");
    }
    
    std::vector<float> output(config.numColumns * config.neuronsPerColumn);
    
    // Copy neuron voltages from device to host
    std::vector<GPUNeuronState> host_neurons(config.numColumns * config.neuronsPerColumn);
    CUDA_CHECK_ERROR(cudaMemcpy(host_neurons.data(), d_neurons, 
                               (config.numColumns * config.neuronsPerColumn) * sizeof(GPUNeuronState), 
                               cudaMemcpyDeviceToHost));
    
    // Extract voltage values
    for (int i = 0; i < (config.numColumns * config.neuronsPerColumn); ++i) {
        output[i] = host_neurons[i].voltage;
    }
    
    return output;
}

void NetworkCUDA::reset() {
    if (!network_initialized) {
        return;
    }
    
    current_time_ms = 0.0f;
    g_stats.reset();
    
    // Reset neuron states
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    
    resetNeuronStatesWrapper(blocks, threads, d_neurons, num_neurons);
    CUDA_KERNEL_CHECK();
}

NetworkStats NetworkCUDA::getStats() const {
    return g_stats;
}

void NetworkCUDA::setLearningRate(float rate) {
    if (rate < 0.0f || rate > 1.0f) {
        throw NetworkException(NetworkError::INVALID_INPUT,
                             "Learning rate must be between 0.0 and 1.0");
    }
    current_learning_rate = rate;
}

void NetworkCUDA::setRewardSignal(float reward) {
    g_stats.current_reward = reward;
}

void NetworkCUDA::enablePlasticity(bool enable) {
    plasticity_enabled = enable;
}

void NetworkCUDA::printNetworkState() const {
    std::cout << "=== NetworkCUDA State ===" << std::endl;
    std::cout << "Neurons: " << (config.numColumns * config.neuronsPerColumn) << std::endl;
    std::cout << "Synapses: " << config.totalSynapses << std::endl;
    std::cout << "Columns: " << config.numColumns << std::endl;
    std::cout << "Neurons per column: " << config.neuronsPerColumn << std::endl;
    std::cout << "Current time: " << current_time_ms << " ms" << std::endl;
    std::cout << "Plasticity enabled: " << (plasticity_enabled ? "Yes" : "No") << std::endl;
    std::cout << "Learning rate: " << current_learning_rate << std::endl;
    std::cout << "Total spikes: " << g_stats.total_spikes << std::endl;
    std::cout << "Average firing rate: " << g_stats.average_firing_rate << " Hz" << std::endl;
    std::cout << "=========================" << std::endl;
}

std::vector<float> NetworkCUDA::getNeuronVoltages() const {
    return getOutput(); // Same implementation for now
}

std::vector<float> NetworkCUDA::getSynapticWeights() const {
    if (!network_initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED,
                             "Network not initialized");
    }
    
    std::vector<float> weights(static_cast<int>(config.totalSynapses));
    
    std::vector<GPUSynapse> host_synapses(static_cast<int>(config.totalSynapses));
    CUDA_CHECK_ERROR(cudaMemcpy(host_synapses.data(), d_synapses,
                               static_cast<int>(config.totalSynapses) * sizeof(GPUSynapse),
                               cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < static_cast<int>(config.totalSynapses); ++i) {
        weights[i] = host_synapses[i].weight;
    }
    
    return weights;
}

// ====================================================================================
// PRIVATE METHODS
// ====================================================================================

void NetworkCUDA::initializeNetwork() {
    std::cout << "[NetworkCUDA] Starting network initialization..." << std::endl;
    
    // Initialize neurons
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    
    std::cout << "[NetworkCUDA] Initializing " << num_neurons << " neurons..." << std::endl;
    initializeNeuronStatesWrapper(blocks, threads, d_neurons, num_neurons);
    CUDA_KERNEL_CHECK();
    
    // Generate and initialize synapses using distance-based connectivity
    int num_synapses = static_cast<int>(config.totalSynapses);
    std::cout << "[NetworkCUDA] Generating " << num_synapses << " synapses..." << std::endl;
    
    if (num_synapses > 0) {
        generateDistanceBasedSynapses();
    } else {
        std::cerr << "[ERROR] No synapses to initialize (totalSynapses = 0)" << std::endl;
    }
    
    // Initialize random states
    initializeRandomStatesWrapper(blocks, threads, d_random_states, 
                                num_neurons, 12345); // seed
    CUDA_KERNEL_CHECK();
    
    std::cout << "[NetworkCUDA] Network initialization complete." << std::endl;
}

void NetworkCUDA::initializeColumns() {
    // Initialize cortical columns if enabled
    if (d_cortical_columns && config.numColumns > 0) {
        initializeCorticalColumnsWrapper(d_cortical_columns, config.numColumns);
        CUDA_KERNEL_CHECK();
    }
}

void NetworkCUDA::generateDistanceBasedSynapses() {
    std::cout << "[NetworkCUDA] Generating distance-based synapses..." << std::endl;
    
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    int num_synapses = static_cast<int>(config.totalSynapses);
    
    if (num_synapses <= 0) {
        std::cerr << "[ERROR] Cannot generate synapses: totalSynapses = " << num_synapses << std::endl;
        return;
    }
    
    // Create host array for synapses
    std::vector<GPUSynapse> host_synapses(num_synapses);
    
    // Generate neuron positions in a 2D grid within each column
    std::vector<std::pair<float, float>> neuron_positions(num_neurons);
    float column_width = config.network_width / config.numColumns;
    float neurons_per_side = std::sqrt(config.neuronsPerColumn);
    
    int neuron_idx = 0;
    for (int col = 0; col < config.numColumns; col++) {
        float col_x_offset = col * column_width;
        
        for (int n = 0; n < config.neuronsPerColumn; n++) {
            float local_x = (n % (int)neurons_per_side) * (column_width / neurons_per_side);
            float local_y = (n / (int)neurons_per_side) * (config.network_height / neurons_per_side);
            
            neuron_positions[neuron_idx] = {col_x_offset + local_x, local_y};
            neuron_idx++;
        }
    }
    
    // Generate synapses based on distance and probability
    int synapse_count = 0;
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(config.wExcMin, config.wExcMax);
    std::uniform_real_distribution<float> delay_dist(config.dMin, config.dMax);
    
    for (int pre = 0; pre < num_neurons && synapse_count < num_synapses; pre++) {
        int connections_from_this_neuron = 0;
        int max_connections = config.localFanOut;
        
        for (int post = 0; post < num_neurons && 
             synapse_count < num_synapses && 
             connections_from_this_neuron < max_connections; post++) {
            
            if (pre == post) continue; // No self-connections
            
            // Calculate distance between neurons
            float dx = neuron_positions[pre].first - neuron_positions[post].first;
            float dy = neuron_positions[pre].second - neuron_positions[post].second;
            float distance = std::sqrt(dx * dx + dy * dy);
            
            // Distance-based connection probability
            float connection_prob = config.connection_probability_base * 
                                  std::exp(-distance / config.distance_decay_constant);
            
            if (prob_dist(rng) < connection_prob) {
                // Create synapse
                GPUSynapse& syn = host_synapses[synapse_count];
                
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(rng);
                syn.delay = delay_dist(rng);
                syn.active = 1;
                
                // Set receptor type (80% excitatory, 20% inhibitory)
                if (prob_dist(rng) < config.exc_ratio) {
                    syn.receptor_index = RECEPTOR_AMPA; // Excitatory
                    syn.weight = std::abs(syn.weight);
                } else {
                    syn.receptor_index = RECEPTOR_GABA_A; // Inhibitory
                    syn.weight = -std::abs(syn.weight); // Negative weight for inhibition
                }
                
                // Initialize plasticity variables
                syn.eligibility_trace = 0.0f;
                syn.dopamine_sensitivity = 1.0f;
                syn.last_pre_spike_time = -1000.0f;
                syn.last_post_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;
                syn.post_compartment = 0; // Somatic compartment
                
                synapse_count++;
                connections_from_this_neuron++;
            }
        }
    }
    
    // Fill remaining synapses with random connections if needed
    while (synapse_count < num_synapses) {
        std::uniform_int_distribution<int> neuron_dist(0, num_neurons - 1);
        int pre = neuron_dist(rng);
        int post = neuron_dist(rng);
        
        if (pre != post && pre >= 0 && pre < num_neurons && post >= 0 && post < num_neurons) {
            GPUSynapse& syn = host_synapses[synapse_count];
            
            syn.pre_neuron_idx = pre;
            syn.post_neuron_idx = post;
            syn.weight = weight_dist(rng);
            syn.delay = delay_dist(rng);
            syn.active = 1;
            syn.receptor_index = (prob_dist(rng) < config.exc_ratio) ? RECEPTOR_AMPA : RECEPTOR_GABA_A;
            
            if (syn.receptor_index == RECEPTOR_GABA_A) {
                syn.weight = -std::abs(syn.weight);
            }
            
            syn.eligibility_trace = 0.0f;
            syn.dopamine_sensitivity = 1.0f;
            syn.last_pre_spike_time = -1000.0f;
            syn.last_post_spike_time = -1000.0f;
            syn.activity_metric = 0.0f;
            syn.post_compartment = 0;
            
            synapse_count++;
        }
    }
    
    // Copy synapses to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_synapses, host_synapses.data(),
                               num_synapses * sizeof(GPUSynapse),
                               cudaMemcpyHostToDevice));
    
    std::cout << "[NetworkCUDA] Generated " << synapse_count << " synapses successfully." << std::endl;
}

void NetworkCUDA::cleanup() {
    if (d_neurons) {
        cudaFree(d_neurons);
        d_neurons = nullptr;
    }
    if (d_synapses) {
        cudaFree(d_synapses);
        d_synapses = nullptr;
    }
    if (d_calcium_levels) {
        cudaFree(d_calcium_levels);
        d_calcium_levels = nullptr;
    }
    if (d_neuron_spike_counts) {
        cudaFree(d_neuron_spike_counts);
        d_neuron_spike_counts = nullptr;
    }
    if (d_random_states) {
        cudaFree(d_random_states);
        d_random_states = nullptr;
    }
    if (d_cortical_columns) {
        cudaFree(d_cortical_columns);
        d_cortical_columns = nullptr;
    }
    if (d_input_currents) {
        cudaFree(d_input_currents);
        d_input_currents = nullptr;
    }
    
    network_initialized = false;
}

void NetworkCUDA::allocateDeviceMemory() {
    try {
        int num_neurons = config.numColumns * config.neuronsPerColumn;
        int num_synapses = static_cast<int>(config.totalSynapses);
        
        std::cout << "[DEBUG] Allocating device memory: " << num_neurons << " neurons, " 
                  << num_synapses << " synapses" << std::endl;
        
        // Allocate neuron memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_neurons, 
                                  num_neurons * sizeof(GPUNeuronState)));
        
        // Test neuron memory allocation
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate neuron memory: " + std::string(cudaGetErrorString(err)));
        }
        
        // Initialize neuron memory to zero
        CUDA_CHECK_ERROR(cudaMemset(d_neurons, 0, num_neurons * sizeof(GPUNeuronState)));
        
        std::cout << "[DEBUG] Neuron memory allocated: " << (num_neurons * sizeof(GPUNeuronState)) << " bytes" << std::endl;
        
        // Allocate synapse memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_synapses, 
                                  num_synapses * sizeof(GPUSynapse)));
        
        // Initialize synapse memory to zero  
        CUDA_CHECK_ERROR(cudaMemset(d_synapses, 0, num_synapses * sizeof(GPUSynapse)));
        
        std::cout << "[DEBUG] Synapse memory allocated: " << (num_synapses * sizeof(GPUSynapse)) << " bytes" << std::endl;
        
        // Allocate calcium levels
        CUDA_CHECK_ERROR(cudaMalloc(&d_calcium_levels, 
                                  num_neurons * sizeof(float)));
        
        // Allocate spike counts
        CUDA_CHECK_ERROR(cudaMalloc(&d_neuron_spike_counts, 
                                  num_neurons * sizeof(int)));
        
        // Allocate random states
        CUDA_CHECK_ERROR(cudaMalloc(&d_random_states, 
                                  num_neurons * sizeof(curandState)));
        
        // Allocate cortical columns (if used)
        if (config.numColumns > 0) {
            CUDA_CHECK_ERROR(cudaMalloc(&d_cortical_columns, 
                                      config.numColumns * sizeof(GPUCorticalColumn)));
        }
        
        // Allocate input currents device memory (maximum expected input size)
        int max_input_size = std::max(64, num_neurons);  // At least 64, up to num_neurons
        CUDA_CHECK_ERROR(cudaMalloc(&d_input_currents, 
                                  max_input_size * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(d_input_currents, 0, max_input_size * sizeof(float)));
        
        std::cout << "[DEBUG] Input currents memory allocated: " << (max_input_size * sizeof(float)) << " bytes" << std::endl;
        
        // Validate all allocations by testing memory access
        std::cout << "[DEBUG] Validating memory allocations..." << std::endl;
        
        // Test neuron memory access
        GPUNeuronState test_neuron;
        test_neuron.voltage = -65.0f;
        test_neuron.spiked = false;
        test_neuron.active = 1;
        
        CUDA_CHECK_ERROR(cudaMemcpy(d_neurons, &test_neuron, sizeof(GPUNeuronState), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(&test_neuron, d_neurons, sizeof(GPUNeuronState), cudaMemcpyDeviceToHost));
        
        if (test_neuron.voltage != -65.0f) {
            throw std::runtime_error("Memory validation failed: neuron memory not accessible");
        }
        
        std::cout << "[DEBUG] Memory validation successful" << std::endl;
        
        // Additional GPU memory access test
        testMemoryAccessWrapper(d_neurons, num_neurons);
        
    } catch (const std::exception& e) {
        cleanup();
        throw;
    }
}

void NetworkCUDA::initializeDeviceArrays() {
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    
    // Zero out allocated memory
    CUDA_CHECK_ERROR(cudaMemset(d_calcium_levels, 0, 
                               num_neurons * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_neuron_spike_counts, 0, 
                               num_neurons * sizeof(int)));
}

void NetworkCUDA::calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const {
    block.x = 256;  // Standard block size
    block.y = 1;
    block.z = 1;
    
    grid.x = (n_elements + block.x - 1) / block.x;
    grid.y = 1;
    grid.z = 1;
}

void NetworkCUDA::updateNeuronsWrapper(float dt_ms) {
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    
    updateNeuronStatesWrapper(blocks, threads, d_neurons, num_neurons, 
                            dt_ms, current_time_ms);
    CUDA_KERNEL_CHECK();
}

void NetworkCUDA::updateSynapsesWrapper(float dt_ms) {
    int num_synapses = static_cast<int>(config.totalSynapses);
    dim3 blocks, threads;
    calculateGridBlockSize(num_synapses, blocks, threads);
    
    updateSynapseStatesWrapper(blocks, threads, d_synapses, num_synapses, dt_ms);
    CUDA_KERNEL_CHECK();
}

void NetworkCUDA::applyPlasticityWrapper(float reward) {
    int num_synapses = static_cast<int>(config.totalSynapses);
    dim3 blocks, threads;
    calculateGridBlockSize(num_synapses, blocks, threads);
    
    applyRewardModulationWrapper(blocks, threads, d_synapses, reward, num_synapses);
    CUDA_KERNEL_CHECK();
}

void NetworkCUDA::processSpikingWrapper() {
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    
    processSpikesWrapper(blocks, threads, d_neurons, d_neuron_spike_counts, 
                       current_time_ms, num_neurons);
    CUDA_KERNEL_CHECK();
}

void NetworkCUDA::updateNetworkStatistics() {
    // Update statistics based on current network state
    // This would typically involve kernels to compute statistics
    #ifdef USE_CUDA
    // For now, simple placeholder implementation
    g_stats.total_spikes += static_cast<int>(current_time_ms * 0.1f); // Mock calculation
    g_stats.average_firing_rate = g_stats.total_spikes / 
                                 (g_stats.total_simulation_time / 1000.0f + 1e-6f);
    #endif
}

void NetworkCUDA::validateConfig() const {
    int num_neurons = config.numColumns * config.neuronsPerColumn;
    int num_synapses = static_cast<int>(config.totalSynapses);
    
    if (num_neurons <= 0 || num_neurons > NetworkConstants::MAX_NEURONS) {
        throw NetworkException(NetworkError::CONFIGURATION_ERROR,
                             "Invalid number of neurons: " + std::to_string(num_neurons));
    }
    
    if (num_synapses <= 0 || num_synapses > NetworkConstants::MAX_SYNAPSES) {
        throw NetworkException(NetworkError::CONFIGURATION_ERROR,
                             "Invalid number of synapses: " + std::to_string(num_synapses));
    }
    
    if (config.numColumns <= 0) {
        throw NetworkException(NetworkError::CONFIGURATION_ERROR,
                             "Invalid number of columns: " + std::to_string(config.numColumns));
    }
    
    if (config.neuronsPerColumn <= 0) {
        throw NetworkException(NetworkError::CONFIGURATION_ERROR,
                             "Invalid neurons per column: " + std::to_string(config.neuronsPerColumn));
    }
}

void NetworkCUDA::checkCudaErrors() const {
    #ifdef USE_CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw NetworkException(NetworkError::CUDA_ERROR,
                             "CUDA error detected: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}