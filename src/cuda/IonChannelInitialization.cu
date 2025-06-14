// IonChannelInitialization.cu
#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>

/**
 * CUDA kernel to initialize ion channel states for all neurons
 * Sets up initial conductances, calcium concentrations, and channel states
 */
__global__ void initializeIonChannels(
    GPUNeuronState* neurons,
    curandState* rng_states,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Initialize random number generator for this neuron
    curandState local_state = rng_states[idx];
    
    // Process each compartment
    for (int c = 0; c < neuron.compartment_count; c++) {
        int comp_type = neuron.compartment_types[c];
        
        // Skip inactive compartments
        if (comp_type == COMPARTMENT_INACTIVE) continue;
        
        // ========================================
        // INITIALIZE CALCIUM DYNAMICS
        // ========================================
        neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
        neuron.ca_buffer[c] = 0.0f;
        
        // Set calcium pump rate based on compartment type
        if (comp_type == COMPARTMENT_SOMA) {
            neuron.ca_pump_rate[c] = CA_EXTRUSION_RATE_SOMA;
        } else {
            neuron.ca_pump_rate[c] = CA_EXTRUSION_RATE_DENDRITE;
        }
        
        // ========================================
        // INITIALIZE SYNAPTIC CHANNEL STATES
        // ========================================
        // AMPA channels - start at rest
        neuron.channels.ampa_g[c] = 0.0f;
        neuron.channels.ampa_state[c] = 0.0f;
        
        // NMDA channels - start at rest
        neuron.channels.nmda_g[c] = 0.0f;
        neuron.channels.nmda_state[c] = 0.0f;
        
        // GABA-A channels - start at rest
        neuron.channels.gaba_a_g[c] = 0.0f;
        neuron.channels.gaba_a_state[c] = 0.0f;
        
        // GABA-B channels - start at rest
        neuron.channels.gaba_b_g[c] = 0.0f;
        neuron.channels.gaba_b_state[c] = 0.0f;
        neuron.channels.gaba_b_g_protein[c] = 0.0f;
        
        // ========================================
        // INITIALIZE VOLTAGE-GATED CHANNEL STATES
        // ========================================
        float v_init = (c == 0) ? neuron.voltage : neuron.voltages[c];
        
        // Calcium channels - initialize based on resting potential
        CaChannel ca_channel = INIT_CA_CHANNEL(comp_type);
        neuron.channels.ca_m[c] = ca_channel.steadyStateActivation(v_init);
        
        // KCa channels - initialize based on resting calcium
        KCaChannel kca_channel = INIT_KCA_CHANNEL(comp_type);
        neuron.channels.kca_m[c] = kca_channel.calciumDependentActivation(RESTING_CA_CONCENTRATION);
        
        // HCN channels - initialize based on resting potential
        HCNChannel hcn_channel = INIT_HCN_CHANNEL(comp_type);
        neuron.channels.hcn_h[c] = hcn_channel.steadyStateActivation(v_init);
        
        // ========================================
        // ADD SMALL RANDOM VARIATIONS
        // ========================================
        // Add 5% random variation to initial states to prevent synchronization
        float noise_scale = 0.05f;
        
        neuron.channels.ca_m[c] *= (1.0f + noise_scale * (curand_uniform(&local_state) - 0.5f));
        neuron.channels.kca_m[c] *= (1.0f + noise_scale * (curand_uniform(&local_state) - 0.5f));
        neuron.channels.hcn_h[c] *= (1.0f + noise_scale * (curand_uniform(&local_state) - 0.5f));
        
        // Clamp to valid ranges
        neuron.channels.ca_m[c] = fmaxf(0.0f, fminf(1.0f, neuron.channels.ca_m[c]));
        neuron.channels.kca_m[c] = fmaxf(0.0f, fminf(1.0f, neuron.channels.kca_m[c]));
        neuron.channels.hcn_h[c] = fmaxf(0.0f, fminf(1.0f, neuron.channels.hcn_h[c]));
        
        // ========================================
        // INITIALIZE LEGACY RECEPTOR CONDUCTANCES
        // ========================================
        // Keep for backward compatibility during transition
        for (int r = 0; r < NUM_RECEPTOR_TYPES; r++) {
            neuron.receptor_conductances[c][r] = 0.0f;
        }
    }
    
    // ========================================
    // INITIALIZE NEURON-LEVEL PARAMETERS
    // ========================================
    
    // Activity and plasticity metrics
    neuron.activity_level = 0.0f;
    neuron.avg_firing_rate = 0.0f;
    neuron.total_excitatory_input = 0.0f;
    neuron.total_inhibitory_input = 0.0f;
    neuron.synaptic_input_rate = 0.0f;
    
    // Spike detection parameters
    neuron.spiked = false;
    neuron.last_spike_time = -1000.0f;  // Long time ago
    neuron.time_since_spike = 1000.0f;
    neuron.spike_count = 0;
    
    // Dendritic spike initialization
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.dendritic_spike[c] = false;
        neuron.dendritic_spike_time[c] = -1000.0f;
        
        // Set dendritic spike thresholds
        if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            neuron.dendritic_threshold[c] = -20.0f;  // Lower threshold for apical dendrites
        } else if (neuron.compartment_types[c] == COMPARTMENT_BASAL) {
            neuron.dendritic_threshold[c] = -15.0f;  // Even lower for basal dendrites
        } else {
            neuron.dendritic_threshold[c] = 0.0f;    // Standard threshold for soma
        }
    }
    
    // Neuromodulation levels (for Phase 4)
    neuron.dopamine_level = 0.0f;
    neuron.serotonin_level = 0.0f;
    neuron.acetylcholine_level = 0.0f;
    neuron.noradrenaline_level = 0.0f;
    
    // Neuromodulator scaling factors (initialize to neutral)
    neuron.neuromod_ampa_scale = 1.0f;
    neuron.neuromod_nmda_scale = 1.0f;
    neuron.neuromod_gaba_scale = 1.0f;
    neuron.neuromod_excitability = 1.0f;
    
    // Plasticity and homeostasis parameters
    neuron.plasticity_threshold = 0.5f;
    neuron.homeostatic_target = 0.1f;  // Target firing rate in Hz
    neuron.metaplasticity_state = 0.0f;
    neuron.developmental_stage = 1;     // Start mature
    
    // Membrane properties (set realistic values)
    neuron.membrane_resistance = 100.0f;    // MΩ
    neuron.membrane_capacitance = 100.0f;   // pF
    
    // Computational efficiency fields
    neuron.last_update_time = 0.0f;
    neuron.needs_update = true;
    neuron.update_priority = 0;
    
    // Monitoring and debugging
    neuron.max_voltage_reached = neuron.voltage;
    neuron.total_current_injected = 0.0f;
    neuron.energy_consumption = 0.0f;
    
    // Update RNG state
    rng_states[idx] = local_state;
}

/**
 * Host function to launch ion channel initialization
 */
void launchIonChannelInitialization(
    GPUNeuronState* d_neurons,
    curandState* d_rng_states,
    int num_neurons
) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    initializeIonChannels<<<grid, block>>>(d_neurons, d_rng_states, num_neurons);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in ion channel initialization: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cudaDeviceSynchronize();
}

/**
 * CUDA kernel to initialize synapses with receptor-specific targeting
 */
__global__ void initializeSynapseReceptors(
    GPUSynapse* synapses,
    curandState* rng_states,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    curandState local_state = rng_states[idx % 1024];  // Reuse RNG states
    
    // ========================================
    // SET RECEPTOR TARGETING
    // ========================================
    
    // Determine receptor type based on synapse type and compartment
    if (synapse.type == 0) {  // Excitatory synapse
        // 70% AMPA, 30% NMDA for excitatory synapses
        if (curand_uniform(&local_state) < 0.7f) {
            synapse.receptor_index = RECEPTOR_AMPA;
            synapse.receptor_weight_fraction = 1.0f;
        } else {
            synapse.receptor_index = RECEPTOR_NMDA;
            synapse.receptor_weight_fraction = 1.0f;
        }
    } else {  // Inhibitory synapse
        // 80% GABA-A, 20% GABA-B for inhibitory synapses
        if (curand_uniform(&local_state) < 0.8f) {
            synapse.receptor_index = RECEPTOR_GABA_A;
            synapse.receptor_weight_fraction = 1.0f;
        } else {
            synapse.receptor_index = RECEPTOR_GABA_B;
            synapse.receptor_weight_fraction = 1.0f;
        }
    }
    
    // ========================================
    // INITIALIZE PLASTICITY PARAMETERS
    // ========================================
    synapse.plasticity_rate = 0.01f;  // Default learning rate
    synapse.meta_weight = 1.0f;       // Default metaplastic weight
    synapse.recent_activity = 0.0f;
    synapse.calcium_trace = 0.0f;
    
    // Enhanced eligibility traces (for Phase 3)
    synapse.eligibility_trace = 0.0f;
    synapse.fast_trace = 0.0f;
    synapse.medium_trace = 0.0f;
    synapse.slow_trace = 0.0f;
    synapse.tag_strength = 0.0f;
    
    // Neuromodulation sensitivity
    synapse.dopamine_sensitivity = 1.0f;
    synapse.plasticity_modulation = 1.0f;
    
    // ========================================
    // INITIALIZE VESICLE DYNAMICS
    // ========================================
    synapse.vesicle_pool_size = 100;   // Total vesicles
    synapse.vesicles_ready = 80;       // Initially 80% ready
    synapse.vesicle_recovery_rate = 0.1f;  // Recovery rate (1/ms)
    
    // Release probability based on synapse type
    if (synapse.type == 0) {  // Excitatory
        synapse.release_probability = 0.3f + 0.2f * curand_uniform(&local_state);  // 0.3-0.5
    } else {  // Inhibitory
        synapse.release_probability = 0.5f + 0.3f * curand_uniform(&local_state);  // 0.5-0.8
    }
    
    // ========================================
    // INITIALIZE ACTIVITY METRICS
    // ========================================
    synapse.activity_metric = 0.0f;
    synapse.last_active = -1000.0f;
    synapse.last_pre_spike_time = -1000.0f;
    synapse.last_post_spike_time = -1000.0f;
    synapse.last_potentiation = -1000.0f;
    synapse.last_depression = -1000.0f;
    
    // ========================================
    // COMPUTATIONAL OPTIMIZATION
    // ========================================
    synapse.needs_plasticity_update = false;
    synapse.last_plasticity_update = 0.0f;
    synapse.plasticity_update_interval = 10;  // Update every 10 timesteps
    
    // Set weight bounds
    synapse.max_weight = 5.0f * synapse.weight;   // 5x initial weight
    synapse.min_weight = 0.1f * synapse.weight;   // 10% of initial weight
    synapse.effective_weight = synapse.weight;
    
    // Plasticity flag
    synapse.is_plastic = true;
}

/**
 * Host function to launch synapse receptor initialization
 */
void launchSynapseReceptorInitialization(
    GPUSynapse* d_synapses,
    curandState* d_rng_states,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    initializeSynapseReceptors<<<grid, block>>>(d_synapses, d_rng_states, num_synapses);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in synapse receptor initialization: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cudaDeviceSynchronize();
}

/**
 * CUDA kernel to validate ion channel initialization
 * Checks for proper initialization and reports any issues
 */
__global__ void validateIonChannelInitialization(
    GPUNeuronState* neurons,
    int* error_count,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    if (neuron.active == 0) return;
    
    bool has_error = false;
    
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Check calcium concentration bounds
        if (neuron.ca_conc[c] < 0.0f || neuron.ca_conc[c] > MAX_CA_CONCENTRATION) {
            has_error = true;
        }
        
        // Check channel state bounds
        if (neuron.channels.ca_m[c] < 0.0f || neuron.channels.ca_m[c] > 1.0f) {
            has_error = true;
        }
        
        if (neuron.channels.kca_m[c] < 0.0f || neuron.channels.kca_m[c] > 1.0f) {
            has_error = true;
        }
        
        if (neuron.channels.hcn_h[c] < 0.0f || neuron.channels.hcn_h[c] > 1.0f) {
            has_error = true;
        }
        
        // Check conductance bounds
        if (neuron.channels.ampa_g[c] < 0.0f || neuron.channels.ampa_g[c] > MAX_CONDUCTANCE) {
            has_error = true;
        }
        
        if (neuron.channels.nmda_g[c] < 0.0f || neuron.channels.nmda_g[c] > MAX_CONDUCTANCE) {
            has_error = true;
        }
        
        if (neuron.channels.gaba_a_g[c] < 0.0f || neuron.channels.gaba_a_g[c] > MAX_CONDUCTANCE) {
            has_error = true;
        }
        
        if (neuron.channels.gaba_b_g[c] < 0.0f || neuron.channels.gaba_b_g[c] > MAX_CONDUCTANCE) {
            has_error = true;
        }
    }
    
    // Check neuron-level parameters
    if (neuron.membrane_capacitance <= 0.0f || neuron.membrane_resistance <= 0.0f) {
        has_error = true;
    }
    
    if (has_error) {
        atomicAdd(error_count, 1);
    }
}

/**
 * Host function to validate initialization
 */
bool validateInitialization(GPUNeuronState* d_neurons, int num_neurons) {
    int* d_error_count;
    int h_error_count = 0;
    
    cudaMalloc(&d_error_count, sizeof(int));
    cudaMemcpy(d_error_count, &h_error_count, sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    validateIonChannelInitialization<<<grid, block>>>(d_neurons, d_error_count, num_neurons);
    
    cudaMemcpy(&h_error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_error_count);
    
    if (h_error_count > 0) {
        fprintf(stderr, "Ion channel initialization validation failed: %d errors found\n", h_error_count);
        return false;
    }
    
    printf("Ion channel initialization validation passed: %d neurons initialized successfully\n", num_neurons);
    return true;
}

/**
 * Host function to print initialization statistics
 */
void printInitializationStats(GPUNeuronState* d_neurons, int num_neurons) {
    // Copy a sample of neurons to host for analysis
    int sample_size = min(100, num_neurons);
    GPUNeuronState* h_neurons = new GPUNeuronState[sample_size];
    
    cudaMemcpy(h_neurons, d_neurons, sample_size * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    printf("\n=== Ion Channel Initialization Statistics ===\n");
    
    // Calculate average calcium concentrations
    float avg_ca_soma = 0.0f, avg_ca_dendrite = 0.0f;
    int soma_count = 0, dendrite_count = 0;
    
    for (int i = 0; i < sample_size; i++) {
        if (h_neurons[i].active == 0) continue;
        
        for (int c = 0; c < h_neurons[i].compartment_count; c++) {
            if (h_neurons[i].compartment_types[c] == COMPARTMENT_SOMA) {
                avg_ca_soma += h_neurons[i].ca_conc[c];
                soma_count++;
            } else if (h_neurons[i].compartment_types[c] != COMPARTMENT_INACTIVE) {
                avg_ca_dendrite += h_neurons[i].ca_conc[c];
                dendrite_count++;
            }
        }
    }
    
    if (soma_count > 0) avg_ca_soma /= soma_count;
    if (dendrite_count > 0) avg_ca_dendrite /= dendrite_count;
    
    printf("Average calcium concentration - Soma: %.6f mM, Dendrites: %.6f mM\n", 
           avg_ca_soma, avg_ca_dendrite);
    
    // Calculate average channel states
    float avg_ca_m = 0.0f, avg_kca_m = 0.0f, avg_hcn_h = 0.0f;
    int total_compartments = 0;
    
    for (int i = 0; i < sample_size; i++) {
        if (h_neurons[i].active == 0) continue;
        
        for (int c = 0; c < h_neurons[i].compartment_count; c++) {
            if (h_neurons[i].compartment_types[c] != COMPARTMENT_INACTIVE) {
                avg_ca_m += h_neurons[i].channels.ca_m[c];
                avg_kca_m += h_neurons[i].channels.kca_m[c];
                avg_hcn_h += h_neurons[i].channels.hcn_h[c];
                total_compartments++;
            }
        }
    }
    
    if (total_compartments > 0) {
        avg_ca_m /= total_compartments;
        avg_kca_m /= total_compartments;
        avg_hcn_h /= total_compartments;
    }
    
    printf("Average channel activation - Ca: %.3f, KCa: %.3f, HCN: %.3f\n", 
           avg_ca_m, avg_kca_m, avg_hcn_h);
    
    printf("=== Initialization Complete ===\n\n");
    
    delete[] h_neurons;
}