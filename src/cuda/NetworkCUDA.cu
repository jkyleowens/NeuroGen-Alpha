#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/CudaUtils.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkPresets.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <NeuroGen/cuda/NeuronUpdateKernel.cuh>
#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/cuda/SynapseInputKernel.cuh>
#include <NeuroGen/cuda/EnhancedSTDPKernel.cuh>
#include <NeuroGen/cuda/EligibilityAndRewardKernels.cuh>
#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>
#include <NeuroGen/cuda/NeuromodulationKernels.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/RandomStateInit.cuh>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Managed memory for global stats, accessible from both CPU and GPU
__managed__ Network::NetworkStats g_stats;

// Global network state (static to this translation unit)
static NetworkConfig g_config;
static GPUNeuronState* d_neurons = nullptr;
static GPUSynapse* d_synapses = nullptr;
static float* d_input_buffer = nullptr;
static float* d_output_buffer = nullptr;
static float* d_global_neuromodulators = nullptr; // GPU pointer for [DA, ACh, 5-HT, NE]
static float h_global_neuromodulators[4] = {0.5f, 0.1f, 0.2f, 0.1f}; // Host-side initial baseline levels

// Network topology tracking
static int total_neurons = 0;
static int total_synapses = 0;
static int input_start_idx, hidden_start_idx, output_start_idx;
static int input_size, hidden_size, output_size;

static float current_time = 0.0f;
static bool network_initialized = false;


// --- Forward Pass and Simulation Loop ---

std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    if (!network_initialized) {
        throw std::runtime_error("Network not initialized. Call initializeNetwork() first.");
    }
    if (input.size() != static_cast<size_t>(g_config.input_size)) {
         throw std::invalid_argument("Input size mismatch in forwardCUDA.");
    }

    safeCudaMemcpy(d_input_buffer, input.data(), input.size(), cudaMemcpyHostToDevice);

    // --- Update Global Neuromodulator Levels (Simplified Model) ---
    // Dopamine (DA) is tied to the external reward signal
    h_global_neuromodulators[0] = h_global_neuromodulators[0] * 0.95f + reward_signal * 0.05f;
    // Acetylcholine (ACh) for attention/plasticity - can be driven by volatility or surprise
    h_global_neuromodulators[1] = h_global_neuromodulators[1] * 0.99f;
    // Serotonin (5-HT) for stability
    h_global_neuromodulators[2] = h_global_neuromodulators[2] * 0.995f;
    safeCudaMemcpy(d_global_neuromodulators, h_global_neuromodulators, 4, cudaMemcpyHostToDevice);

    const float simulation_time_ms = 10.0f;
    const int simulation_steps = static_cast<int>(simulation_time_ms / g_config.dt);

    dim3 block_dim = getOptimalBlockSize();
    dim3 neuron_grid_dim = getOptimalGridSize(total_neurons);
    dim3 synapse_grid_dim = getOptimalGridSize(total_synapses);
    dim3 input_grid_dim = getOptimalGridSize(g_config.input_size);

    // --- Core Simulation Loop ---
    for (int step = 0; step < simulation_steps; ++step) {
        current_time += g_config.dt;

        // STEP 1: Apply Neuromodulatory Effects
        // This sets the network's "state" for the current timestep, altering neuronal
        // and synaptic properties before they are processed.
        applyNeuromodulationToNeuronsKernel<<<neuron_grid_dim, block_dim>>>(d_neurons, d_global_neuromodulators, total_neurons);
        applyNeuromodulationToSynapsesKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, d_global_neuromodulators, total_synapses);
        CUDA_CHECK_KERNEL();

        // STEP 2: Inject External Current
        if (step % 5 == 0) {
            injectInputCurrentImproved<<<input_grid_dim, block_dim>>>(
                d_neurons + input_start_idx, d_input_buffer, g_config.input_size,
                current_time, g_config.input_current_scale
            );
        }

        // STEP 3: Update Neuron States
        // The core RK4 kernel computes all intrinsic and synaptic currents to update voltages.
        enhancedRK4NeuronUpdateKernel<<<neuron_grid_dim, block_dim>>>(d_neurons, g_config.dt, current_time, total_neurons);
        CUDA_CHECK_KERNEL();

        // STEP 4: Detect Spikes
        resetSpikeFlags<<<neuron_grid_dim, block_dim>>>(d_neurons, total_neurons);
        launchSpikeDetectionKernel<<<neuron_grid_dim, block_dim>>>(d_neurons, g_config.spike_threshold, total_neurons, current_time);
        CUDA_CHECK_KERNEL();

        // STEP 5: Process Synaptic Transmission
        // Spikes from step 4 are transmitted, affecting postsynaptic receptor states for the *next* cycle.
        if (total_synapses > 0) {
            synapseInputKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, d_neurons, total_synapses);
            CUDA_CHECK_KERNEL();
        }
    }

    // --- Extract Output ---
    std::vector<float> final_output(g_config.output_size);
    dim3 output_grid_dim = getOptimalGridSize(g_config.output_size);
    extractOutputImproved<<<output_grid_dim, block_dim>>>(
        d_neurons + output_start_idx, d_output_buffer, g_config.output_size, current_time
    );
    safeCudaMemcpy(final_output.data(), d_output_buffer, g_config.output_size, cudaMemcpyDeviceToHost);

    return NetworkCUDAInternal::applySoftmax(final_output);
}


// --- Learning and Weight Updates ---
void updateSynapticWeightsCUDA(float reward_signal) {
    if (!network_initialized || total_synapses == 0) return;

    dim3 synapse_grid_dim = getOptimalGridSize(total_synapses);
    dim3 neuron_grid_dim = getOptimalGridSize(total_neurons);
    dim3 block_dim = getOptimalBlockSize();
    float dt = g_config.dt;

    // --- BIOLOGICALLY-INSPIRED LEARNING & STABILIZATION SEQUENCE ---

    // STEP 1: Update Eligibility Traces (Passive Decay & Consolidation)
    eligibilityTraceUpdateKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, dt, total_synapses);
    CUDA_CHECK_KERNEL();

    // STEP 2: Compute Plasticity Potential (Calcium & Spike-Timing Dependent)
    // This kernel "tags" synapses for change based on local activity.
    enhancedSTDPKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, d_neurons, current_time, dt, total_synapses);
    CUDA_CHECK_KERNEL();

    // STEP 3: Apply Reward-Modulated Consolidation (Reinforcement)
    // The global reward signal converts tagged potential into lasting weight changes.
    rewardModulationKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, reward_signal, dt, total_synapses);
    CUDA_CHECK_KERNEL();

    // STEP 4: Apply Homeostatic Plasticity (Stability)
    // This runs periodically to keep the network from becoming over- or under-active.
    static int update_counter = 0;
    if (++update_counter % 100 == 0) {
        computeSynapticScalingFactorKernel<<<neuron_grid_dim, block_dim>>>(d_neurons, dt * 100, total_neurons);
        applySynapticScalingKernel<<<synapse_grid_dim, block_dim>>>(d_synapses, d_neurons, total_synapses);
        CUDA_CHECK_KERNEL();
    }
}


// --- Initialization and Cleanup ---

void initializeNetwork() {
    if (network_initialized) return;
    std::cout << "[INIT] Starting CUDA network initialization..." << std::endl;

    g_config = NetworkPresets::trading_optimized();
    g_config.print();

    total_neurons = g_config.input_size + g_config.hidden_size + g_config.output_size;
    input_size = g_config.input_size;
    hidden_size = g_config.hidden_size;
    output_size = g_config.output_size;
    input_start_idx = 0;
    hidden_start_idx = input_size;
    output_start_idx = input_size + hidden_size;

    // Allocate GPU memory for all network components
    safeCudaMalloc(&d_neurons, total_neurons);
    safeCudaMalloc(&d_input_buffer, input_size);
    safeCudaMalloc(&d_output_buffer, output_size);
    safeCudaMalloc(&d_global_neuromodulators, 4);
    safeCudaMemcpy(d_global_neuromodulators, h_global_neuromodulators, 4, cudaMemcpyHostToDevice);

    std::vector<GPUNeuronState> host_neurons(total_neurons);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> voltage_dist(RESTING_POTENTIAL, 5.0f);
    
    for (int i = 0; i < total_neurons; ++i) {
        auto& n = host_neurons[i];
        n.active = 1;
        n.voltage = voltage_dist(gen);
        n.spiked = false;
        n.last_spike_time = -1000.0f;
        n.spike_threshold = g_config.spike_threshold;
        n.membrane_capacitance = 1.0f;
        n.compartment_count = MAX_COMPARTMENTS; // Set to max for simplicity
        n.avg_firing_rate = 0.0f;
        n.homeostatic_scaling_factor = 1.0f;

        // Initialize all compartments and channels to resting states
        for(int c=0; c < MAX_COMPARTMENTS; ++c) {
             n.voltages[c] = n.voltage;
             n.m_comp[c] = 0.05f; n.h_comp[c] = 0.6f; n.n_comp[c] = 0.32f;
             n.channels.ampa_g[c] = 0.0f; n.channels.ampa_state[c] = 0.0f;
             n.channels.nmda_g[c] = 0.0f; n.channels.nmda_state[c] = 0.0f;
             n.channels.gaba_a_g[c] = 0.0f; n.channels.gaba_a_state[c] = 0.0f;
             n.channels.gaba_b_g[c] = 0.0f; n.channels.gaba_b_state[c] = 0.0f;
             n.ca_conc[c] = RESTING_CA_CONCENTRATION;
        }
    }
    safeCudaMemcpy(d_neurons, host_neurons.data(), total_neurons, cudaMemcpyHostToDevice);
    
    std::vector<GPUSynapse> host_synapses;
    NetworkCUDAInternal::createNetworkTopology(host_synapses, gen);
    total_synapses = host_synapses.size();
    
    if (total_synapses > 0) {
        safeCudaMalloc(&d_synapses, total_synapses);
        safeCudaMemcpy(d_synapses, host_synapses.data(), total_synapses, cudaMemcpyHostToDevice);
    }
    
    network_initialized = true;
    std::cout << "[CUDA] Network initialization complete." << std::endl;
}

void cleanupNetwork() {
    std::cout << "[CLEANUP] Cleaning up CUDA network..." << std::endl;
    if(d_neurons) cudaFree(d_neurons);
    if(d_synapses) cudaFree(d_synapses);
    if(d_input_buffer) cudaFree(d_input_buffer);
    if(d_output_buffer) cudaFree(d_output_buffer);
    if(d_global_neuromodulators) cudaFree(d_global_neuromodulators);
    
    d_neurons = nullptr;
    d_synapses = nullptr;
    d_input_buffer = nullptr;
    d_output_buffer = nullptr;
    d_global_neuromodulators = nullptr;
    network_initialized = false;
    std::cout << "[CLEANUP] Network cleanup complete." << std::endl;
}

// ... (The NetworkCUDAInternal namespace and its functions like createNetworkTopology and applySoftmax should be included here as they were in the original file)