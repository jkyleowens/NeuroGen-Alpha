// IntegratedSimulationLoop.h
#ifndef INTEGRATED_SIMULATION_LOOP_H
#define INTEGRATED_SIMULATION_LOOP_H

#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Enhanced Network CUDA class with comprehensive ion channel dynamics
 * Integrates all Phase 2 enhancements into the existing framework
 */
class EnhancedNetworkCUDA {
public:
    EnhancedNetworkCUDA(int num_neurons, int num_synapses);
    ~EnhancedNetworkCUDA();
    
    // ========================================
    // INITIALIZATION METHODS
    // ========================================
    bool initialize();
    bool initializeIonChannels();
    bool initializeSynapseReceptors();
    bool validateInitialization();
    
    // ========================================
    // SIMULATION STEP METHODS
    // ========================================
    void simulationStep(float dt, float current_time);
    void updateNeuronDynamics(float dt, float current_time);
    void processSynapticInput(float current_time, float dt);
    void updateCalciumDynamics(float dt);
    void injectBackgroundNoise(float noise_rate, float dt);
    
    // ========================================
    // MONITORING AND ANALYSIS
    // ========================================
    void getNetworkStatistics(NetworkStats* stats);
    void getCalciumStatistics(CalciumStats* stats);
    void getSynapticStatistics(SynapticStats* stats);
    
    // ========================================
    // DEVICE MEMORY ACCESS
    // ========================================
    GPUNeuronState* getDeviceNeurons() const { return d_neurons_; }
    GPUSynapse* getDeviceSynapses() const { return d_synapses_; }
    GPUSpikeEvent* getDeviceSpikeEvents() const { return d_spike_events_; }
    
    // ========================================
    // CONFIGURATION METHODS
    // ========================================
    void setBackgroundNoiseRate(float rate) { background_noise_rate_ = rate; }
    void setCalciumDiffusionEnabled(bool enabled) { calcium_diffusion_enabled_ = enabled; }
    void setDetailedMonitoringEnabled(bool enabled) { detailed_monitoring_enabled_ = enabled; }
    
private:
    // ========================================
    // DEVICE MEMORY POINTERS
    // ========================================
    GPUNeuronState* d_neurons_;
    GPUSynapse* d_synapses_;
    GPUSpikeEvent* d_spike_events_;
    curandState* d_rng_states_;
    
    // Auxiliary arrays for computations
    float* d_excitatory_currents_;
    float* d_inhibitory_currents_;
    int* d_synapse_to_neuron_map_;
    int* d_neuron_synapse_counts_;
    
    // ========================================
    // SIMULATION PARAMETERS
    // ========================================
    int num_neurons_;
    int num_synapses_;
    int max_spike_events_;
    float background_noise_rate_;
    bool calcium_diffusion_enabled_;
    bool detailed_monitoring_enabled_;
    
    // Performance monitoring
    float last_simulation_time_;
    int simulation_step_count_;
    
    // ========================================
    // HELPER METHODS
    // ========================================
    bool allocateDeviceMemory();
    void deallocateDeviceMemory();
    bool buildSynapseMapping();
    void resetSpikeEvents();
    void detectSpikes(float current_time);
};

/**
 * Statistics structures for monitoring
 */
struct NetworkStats {
    float avg_firing_rate;
    float avg_membrane_potential;
    float total_excitatory_current;
    float total_inhibitory_current;
    int active_neurons;
    int total_spikes;
    float simulation_time_ms;
};

struct CalciumStats {
    float avg_soma_calcium;
    float avg_dendrite_calcium;
    float max_calcium_level;
    int overflow_neurons;
    float avg_buffer_occupancy;
};

struct SynapticStats {
    float avg_synaptic_weight;
    float avg_release_probability;
    int active_synapses;
    float avg_vesicle_availability;
    float total_neurotransmitter_release;
};

// ========================================
// IMPLEMENTATION
// ========================================

EnhancedNetworkCUDA::EnhancedNetworkCUDA(int num_neurons, int num_synapses)
    : num_neurons_(num_neurons)
    , num_synapses_(num_synapses)
    , max_spike_events_(num_neurons * 10)  // Allow up to 10 spikes per neuron per timestep
    , background_noise_rate_(5.0f)  // 5 Hz default
    , calcium_diffusion_enabled_(true)
    , detailed_monitoring_enabled_(false)
    , last_simulation_time_(0.0f)
    , simulation_step_count_(0)
    , d_neurons_(nullptr)
    , d_synapses_(nullptr)
    , d_spike_events_(nullptr)
    , d_rng_states_(nullptr)
    , d_excitatory_currents_(nullptr)
    , d_inhibitory_currents_(nullptr)
    , d_synapse_to_neuron_map_(nullptr)
    , d_neuron_synapse_counts_(nullptr)
{
}

EnhancedNetworkCUDA::~EnhancedNetworkCUDA() {
    deallocateDeviceMemory();
}

bool EnhancedNetworkCUDA::initialize() {
    printf("Initializing Enhanced Network CUDA with %d neurons and %d synapses...\n", 
           num_neurons_, num_synapses_);
    
    // Allocate device memory
    if (!allocateDeviceMemory()) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return false;
    }
    
    // Initialize ion channels
    if (!initializeIonChannels()) {
        fprintf(stderr, "Failed to initialize ion channels\n");
        return false;
    }
    
    // Initialize synapse receptors
    if (!initializeSynapseReceptors()) {
        fprintf(stderr, "Failed to initialize synapse receptors\n");
        return false;
    }
    
    // Build synapse mapping for efficient processing
    if (!buildSynapseMapping()) {
        fprintf(stderr, "Failed to build synapse mapping\n");
        return false;
    }
    
    // Validate initialization
    if (!validateInitialization()) {
        fprintf(stderr, "Initialization validation failed\n");
        return false;
    }
    
    printf("Enhanced Network CUDA initialization complete\n");
    return true;
}

void EnhancedNetworkCUDA::simulationStep(float dt, float current_time) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // ========================================
    // STEP 1: PROCESS SYNAPTIC INPUT
    // ========================================
    processSynapticInput(current_time, dt);
    
    // ========================================
    // STEP 2: UPDATE CALCIUM DYNAMICS
    // ========================================
    if (calcium_diffusion_enabled_) {
        updateCalciumDynamics(dt);
    }
    
    // ========================================
    // STEP 3: UPDATE NEURON DYNAMICS
    // ========================================
    updateNeuronDynamics(dt, current_time);
    
    // ========================================
    // STEP 4: DETECT AND RECORD SPIKES
    // ========================================
    detectSpikes(current_time);
    
    // ========================================
    // STEP 5: INJECT BACKGROUND NOISE
    // ========================================
    if (background_noise_rate_ > 0.0f) {
        injectBackgroundNoise(background_noise_rate_, dt);
    }
    
    // ========================================
    // STEP 6: UPDATE PERFORMANCE METRICS
    // ========================================
    simulation_step_count_++;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    last_simulation_time_ = duration.count() / 1000.0f;  // Convert to milliseconds
    
    // Synchronize to ensure all kernels complete
    cudaDeviceSynchronize();
}

void EnhancedNetworkCUDA::processSynapticInput(float current_time, float dt) {
    // Launch enhanced synaptic input processing
    launchEnhancedSynapticInput(
        d_synapses_,
        d_neurons_,
        d_spike_events_,
        d_excitatory_currents_,
        d_inhibitory_currents_,
        d_synapse_to_neuron_map_,
        d_neuron_synapse_counts_,
        num_synapses_,
        num_neurons_,
        max_spike_events_,  // Current number of spike events
        current_time,
        dt
    );
}

void EnhancedNetworkCUDA::updateCalciumDynamics(float dt) {
    // Launch calcium diffusion and related processes
    launchCalciumDynamics(d_neurons_, dt, num_neurons_);
}

void EnhancedNetworkCUDA::updateNeuronDynamics(float dt, float current_time) {
    // Launch enhanced RK4 neuron update kernel
    dim3 block(256);
    dim3 grid((num_neurons_ + block.x - 1) / block.x);
    
    enhancedRK4NeuronUpdateKernel<<<grid, block>>>(
        d_neurons_, dt, current_time, num_neurons_
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in neuron dynamics update: %s\n", cudaGetErrorString(err));
    }
}

void EnhancedNetworkCUDA::injectBackgroundNoise(float noise_rate, float dt) {
    injectSynapticNoise(d_neurons_, d_rng_states_, noise_rate, dt, num_neurons_);
}

void EnhancedNetworkCUDA::detectSpikes(float current_time) {
    // This kernel would detect spikes and add them to the spike event buffer
    // Implementation depends on the existing spike detection mechanism
    
    dim3 block(256);
    dim3 grid((num_neurons_ + block.x - 1) / block.x);
    
    // Launch spike detection kernel (existing or new implementation)
    // detectSpikesKernel<<<grid, block>>>(d_neurons_, d_spike_events_, current_time, num_neurons_);
}

bool EnhancedNetworkCUDA::allocateDeviceMemory() {
    cudaError_t err;
    
    // Allocate neuron array
    err = cudaMalloc(&d_neurons_, num_neurons_ * sizeof(GPUNeuronState));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate neuron memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate synapse array
    err = cudaMalloc(&d_synapses_, num_synapses_ * sizeof(GPUSynapse));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate synapse memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate spike events array
    err = cudaMalloc(&d_spike_events_, max_spike_events_ * sizeof(GPUSpikeEvent));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate spike events memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate RNG states
    err = cudaMalloc(&d_rng_states_, num_neurons_ * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate RNG states memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate auxiliary arrays
    err = cudaMalloc(&d_excitatory_currents_, num_neurons_ * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate excitatory currents memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_inhibitory_currents_, num_neurons_ * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate inhibitory currents memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_synapse_to_neuron_map_, num_synapses_ * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate synapse mapping memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_neuron_synapse_counts_, num_neurons_ * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate neuron synapse counts memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    printf("Device memory allocation successful\n");
    return true;
}

bool EnhancedNetworkCUDA::initializeIonChannels() {
    // Initialize RNG states first
    dim3 block(256);
    dim3 grid((num_neurons_ + block.x - 1) / block.x);
    
    // Initialize RNG states kernel (would need to be implemented)
    // initRNGStates<<<grid, block>>>(d_rng_states_, time(NULL), num_neurons_);
    
    // Launch ion channel initialization
    launchIonChannelInitialization(d_neurons_, d_rng_states_, num_neurons_);
    
    printf("Ion channel initialization complete\n");
    return true;
}

bool EnhancedNetworkCUDA::initializeSynapseReceptors() {
    // Launch synapse receptor initialization
    launchSynapseReceptorInitialization(d_synapses_, d_rng_states_, num_synapses_);
    
    printf("Synapse receptor initialization complete\n");
    return true;
}

bool EnhancedNetworkCUDA::validateInitialization() {
    // Validate ion channel initialization
    bool ion_channels_valid = validateInitialization(d_neurons_, num_neurons_);
    
    if (!ion_channels_valid) {
        fprintf(stderr, "Ion channel validation failed\n");
        return false;
    }
    
    // Additional validation for synapses could be added here
    
    printf("Initialization validation passed\n");
    return true;
}

void EnhancedNetworkCUDA::getNetworkStatistics(NetworkStats* stats) {
    // Copy neurons to host for analysis (sample-based for performance)
    int sample_size = min(1000, num_neurons_);
    GPUNeuronState* h_neurons = new GPUNeuronState[sample_size];
    
    cudaMemcpy(h_neurons, d_neurons_, sample_size * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float total_firing_rate = 0.0f;
    float total_voltage = 0.0f;
    float total_exc_current = 0.0f;
    float total_inh_current = 0.0f;
    int active_count = 0;
    int total_spikes = 0;
    
    for (int i = 0; i < sample_size; i++) {
        if (h_neurons[i].active) {
            active_count++;
            total_firing_rate += h_neurons[i].avg_firing_rate;
            total_voltage += h_neurons[i].voltage;
            total_exc_current += h_neurons[i].total_excitatory_input;
            total_inh_current += h_neurons[i].total_inhibitory_input;
            total_spikes += h_neurons[i].spike_count;
        }
    }
    
    if (active_count > 0) {
        stats->avg_firing_rate = total_firing_rate / active_count;
        stats->avg_membrane_potential = total_voltage / active_count;
        stats->total_excitatory_current = total_exc_current / active_count;
        stats->total_inhibitory_current = total_inh_current / active_count;
    } else {
        stats->avg_firing_rate = 0.0f;
        stats->avg_membrane_potential = 0.0f;
        stats->total_excitatory_current = 0.0f;
        stats->total_inhibitory_current = 0.0f;
    }
    
    stats->active_neurons = active_count;
    stats->total_spikes = total_spikes;
    stats->simulation_time_ms = last_simulation_time_;
    
    delete[] h_neurons;
}

void EnhancedNetworkCUDA::getCalciumStatistics(CalciumStats* stats) {
    getCalciumStatistics(
        d_neurons_, num_neurons_,
        &stats->avg_soma_calcium,
        &stats->avg_dendrite_calcium,
        &stats->max_calcium_level,
        &stats->overflow_neurons
    );
    
    // Calculate buffer occupancy (sample-based)
    int sample_size = min(100, num_neurons_);
    GPUNeuronState* h_neurons = new GPUNeuronState[sample_size];
    cudaMemcpy(h_neurons, d_neurons_, sample_size * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    float total_buffer_occupancy = 0.0f;
    int compartment_count = 0;
    
    for (int i = 0; i < sample_size; i++) {
        if (h_neurons[i].active) {
            for (int c = 0; c < h_neurons[i].compartment_count; c++) {
                if (h_neurons[i].compartment_types[c] != COMPARTMENT_INACTIVE) {
                    total_buffer_occupancy += h_neurons[i].ca_buffer[c] / CA_BUFFER_CAPACITY;
                    compartment_count++;
                }
            }
        }
    }
    
    stats->avg_buffer_occupancy = (compartment_count > 0) ? 
                                 total_buffer_occupancy / compartment_count : 0.0f;
    
    delete[] h_neurons;
}

void EnhancedNetworkCUDA::deallocateDeviceMemory() {
    if (d_neurons_) cudaFree(d_neurons_);
    if (d_synapses_) cudaFree(d_synapses_);
    if (d_spike_events_) cudaFree(d_spike_events_);
    if (d_rng_states_) cudaFree(d_rng_states_);
    if (d_excitatory_currents_) cudaFree(d_excitatory_currents_);
    if (d_inhibitory_currents_) cudaFree(d_inhibitory_currents_);
    if (d_synapse_to_neuron_map_) cudaFree(d_synapse_to_neuron_map_);
    if (d_neuron_synapse_counts_) cudaFree(d_neuron_synapse_counts_);
}

#endif // INTEGRATED_SIMULATION_LOOP_H