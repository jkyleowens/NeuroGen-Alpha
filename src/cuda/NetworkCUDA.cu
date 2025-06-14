#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/CudaUtils.h>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/NeuronInitialization.cuh>
#include <NeuroGen/cuda/SynapseInputKernel.cuh>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/cuda/HebbianLearningKernel.cuh>
#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>
#include <NeuroGen/cuda/StructuralPlasticityKernels.cuh>
#include <NeuroGen/cuda/EligibilityTraceKernel.cuh>
#include <NeuroGen/cuda/RewardModulationKernel.cuh>
#include <NeuroGen/cuda/RandomStateInit.cuh>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/cuda/CorticalColumn.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <memory>

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
    current_time_ms(0.0f)
{
    std::cout << "Initializing NetworkCUDA..." << std::endl;
    try {
        initializeNetwork();
        initializeColumns();
    } catch (const std::exception& e) {
        std::cerr << "Exception during NetworkCUDA initialization: " << e.what() << std::endl;
        cleanup();
        throw;
    }
    std::cout << "NetworkCUDA initialized successfully." << std::endl;
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
    current_time_ms += dt_ms;

    // Kernel launch configurations
    int num_neurons = config.num_neurons;
    int num_synapses = config.num_synapses;

    dim3 neuron_blocks, neuron_threads;
    calculateGridBlockSize(num_neurons, neuron_blocks, neuron_threads);

    dim3 synapse_blocks, synapse_threads;
    calculateGridBlockSize(num_synapses, synapse_blocks, synapse_threads);

    // 1. Apply external input currents
    applyInputCurrentsWrapper(d_neurons, input_currents.data(), num_neurons);

    // 2. Process synaptic inputs
    processSynapticInputsWrapper(d_neurons, d_synapses, num_synapses);

    // 3. Update neuron states (voltage, recovery, etc.)
    updateNeuronStateWrapper(neuron_blocks, neuron_threads, d_neurons, d_calcium_levels, dt_ms, num_neurons);
    
    // 4. Check for and process neuron spikes
    processSpikesWrapper(neuron_blocks, neuron_threads, d_neurons, d_neuron_spike_counts, current_time_ms, num_neurons);

    // 5. Update eligibility traces
    updateEligibilityTracesWrapper(synapse_blocks, synapse_threads, d_synapses, d_neurons, dt_ms, num_synapses);
    
    // 6. Apply reward modulation
    applyRewardModulationWrapper(synapse_blocks, synapse_threads, d_synapses, reward, num_synapses);

    // 7. Apply Hebbian learning (STDP)
    applyHebbianLearningWrapper(synapse_blocks, synapse_threads, d_synapses, d_neurons, num_synapses);

    // 8. Homeostatic mechanisms (run less frequently)
    if (static_cast<int>(current_time_ms / dt_ms) % 10 == 0) {
        applyHomeostaticMechanismsWrapper(neuron_blocks, neuron_threads, d_neurons, d_synapses, dt_ms, num_neurons, num_synapses);
    }
    
    // 9. Structural plasticity (run even less frequently)
    if (static_cast<int>(current_time_ms / dt_ms) % 100 == 0) {
        applyStructuralPlasticityWrapper(d_synapses, d_neurons, d_random_states, num_synapses, num_neurons);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NetworkCUDA::getSpikeCounts(std::vector<int>& spike_counts) {
    if (spike_counts.size() != config.num_neurons) {
        spike_counts.resize(config.num_neurons);
    }
    CUDA_CHECK(cudaMemcpy(spike_counts.data(), d_neuron_spike_counts, config.num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
}

void NetworkCUDA::resetSpikeCounts() {
    CUDA_CHECK(cudaMemset(d_neuron_spike_counts, 0, config.num_neurons * sizeof(int)));
}

// ====================================================================================
// PRIVATE HELPER METHODS
// ====================================================================================

void NetworkCUDA::initializeNetwork() {
    std::cout << "Allocating memory for " << config.num_neurons << " neurons and " << config.num_synapses << " synapses." << std::endl;

    CUDA_CHECK(cudaMalloc(&d_neurons, config.num_neurons * sizeof(GPUNeuronState)));
    CUDA_CHECK(cudaMalloc(&d_synapses, config.num_synapses * sizeof(GPUSynapse)));
    CUDA_CHECK(cudaMalloc(&d_calcium_levels, config.num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_neuron_spike_counts, config.num_neurons * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_random_states, config.num_neurons * sizeof(curandState)));
    
    dim3 blocks, threads;
    calculateGridBlockSize(config.num_neurons, blocks, threads);
    initializeNeuronsWrapper(blocks, threads, d_neurons, config.num_neurons);
    initializeRandomStatesWrapper(blocks, threads, d_random_states, time(0), config.num_neurons);
    
    std::cout << "GPU memory allocated and neurons initialized." << std::endl;
}

void NetworkCUDA::initializeColumns() {
    if (config.num_neurons > 0) {
        h_cortical_columns.emplace_back(0, config.num_neurons - 1);
        
        CUDA_CHECK(cudaMalloc(&d_cortical_columns, h_cortical_columns.size() * sizeof(GPUCorticalColumn)));
        CUDA_CHECK(cudaMemcpy(d_cortical_columns, h_cortical_columns.data(), h_cortical_columns.size() * sizeof(GPUCorticalColumn), cudaMemcpyHostToDevice));
        std::cout << "Cortical columns initialized and copied to GPU." << std::endl;
    }
}

void NetworkCUDA::cleanup() {
    CUDA_CHECK(cudaFree(d_neurons));
    CUDA_CHECK(cudaFree(d_synapses));
    CUDA_CHECK(cudaFree(d_calcium_levels));
    CUDA_CHECK(cudaFree(d_neuron_spike_counts));
    CUDA_CHECK(cudaFree(d_random_states));
    CUDA_CHECK(cudaFree(d_cortical_columns));
}

void NetworkCUDA::calculateGridBlockSize(int N, dim3& blocks, dim3& threads) {
    threads.x = 256;
    blocks.x = (N + threads.x - 1) / threads.x;
}