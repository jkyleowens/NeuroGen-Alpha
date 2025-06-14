#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

/**
 * CUDA kernel for calcium diffusion between compartments
 * Implements realistic calcium dynamics with buffering, diffusion, and extrusion
 */
__global__ void calciumDiffusionKernel(
    GPUNeuronState* neurons,
    float dt,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];

    // Skip inactive neurons
    if (neuron.active == 0) return;

    // Temporary arrays to store new calcium concentrations
    float new_ca_conc[MAX_COMPARTMENTS];
    float new_ca_buffer[MAX_COMPARTMENTS];

    // Copy current calcium concentrations
    for (int c = 0; c < neuron.compartment_count; c++) {
        new_ca_conc[c] = neuron.ca_conc[c];
        new_ca_buffer[c] = neuron.ca_buffer[c];
    }

    // ========================================
    // COMPUTE CALCIUM DIFFUSION
    // ========================================
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;

        float ca_current = neuron.ca_conc[c];
        float diffusion_flux = 0.0f;

        // ========================================
        // DIFFUSION TO/FROM PARENT COMPARTMENT
        // ========================================
        if (c > 0) {  // Not soma
            int parent = neuron.parent_compartment[c];
            if (parent >= 0 && parent < neuron.compartment_count) {
                float parent_ca = neuron.ca_conc[parent];
                float concentration_gradient = parent_ca - ca_current;

                // Diffusion rate depends on compartment coupling
                float diffusion_conductance = neuron.coupling_conductance[c] * CA_DIFFUSION_RATE;
                diffusion_flux += diffusion_conductance * concentration_gradient;
            }
        }

        // ========================================
        // DIFFUSION TO/FROM CHILD COMPARTMENTS
        // ========================================
        if (c == 0) {  // Soma - check all children
            for (int child = 1; child < neuron.compartment_count; child++) {
                if (neuron.parent_compartment[child] == c) {
                    float child_ca = neuron.ca_conc[child];
                    float concentration_gradient = child_ca - ca_current;

                    float diffusion_conductance = neuron.coupling_conductance[child] * CA_DIFFUSION_RATE;
                    diffusion_flux -= diffusion_conductance * concentration_gradient;  // Negative because flux goes to child
                }
            }
        }

        // ========================================
        // CALCIUM EXTRUSION (PUMPS AND EXCHANGERS)
        // ========================================
        float ca_extrusion = neuron.ca_pump_rate[c] * (ca_current - RESTING_CA_CONCENTRATION);

        // Cooperative calcium extrusion (Hill equation)
        float hill_coef = 2.0f;
        float pump_km = 0.001f;  // mM
        float max_pump_rate = neuron.ca_pump_rate[c] * 2.0f;

        float cooperative_extrusion = max_pump_rate *
            powf(ca_current, hill_coef) / (powf(pump_km, hill_coef) + powf(ca_current, hill_coef));

        // Total extrusion
        float total_extrusion = ca_extrusion + cooperative_extrusion;

        // ========================================
        // CALCIUM BUFFERING DYNAMICS
        // ========================================
        float buffer_current = neuron.ca_buffer[c];

        // Binding kinetics (simple mass action)
        float kon = 100.0f;     // Forward binding rate (1/mM/ms)
        float koff = 1.0f;      // Reverse binding rate (1/ms)

        float free_buffer = CA_BUFFER_CAPACITY - buffer_current;
        float binding_rate = kon * ca_current * free_buffer;
        float unbinding_rate = koff * buffer_current;

        float net_buffering = binding_rate - unbinding_rate;

        // ========================================
        // CALCIUM LEAK FROM INTRACELLULAR STORES
        // ========================================
        float store_leak = 0.0f;

        // ER calcium leak (compartment-specific)
        if (neuron.compartment_types[c] == COMPARTMENT_SOMA) {
            store_leak = 0.001f;  // Small leak in soma
        } else if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            store_leak = 0.002f;  // Larger leak in apical dendrites
        } else {
            store_leak = 0.0005f; // Minimal leak in basal dendrites
        }

        // ========================================
        // UPDATE CALCIUM CONCENTRATION
        // ========================================
        float dca_dt = diffusion_flux - total_extrusion - net_buffering + store_leak;
        new_ca_conc[c] = ca_current + dca_dt * dt;

        // Update buffer concentration
        float dbuffer_dt = net_buffering;
        new_ca_buffer[c] = buffer_current + dbuffer_dt * dt;

        // ========================================
        // ENFORCE BOUNDS AND STABILITY
        // ========================================
        new_ca_conc[c] = fmaxf(MIN_CA_CONCENTRATION,
                               fminf(MAX_CA_CONCENTRATION, new_ca_conc[c]));
        new_ca_buffer[c] = fmaxf(0.0f,
                                fminf(CA_BUFFER_CAPACITY, new_ca_buffer[c]));
    }

    // ========================================
    // COPY BACK NEW CONCENTRATIONS
    // ========================================
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] != COMPARTMENT_INACTIVE) {
            neuron.ca_conc[c] = new_ca_conc[c];
            neuron.ca_buffer[c] = new_ca_buffer[c];
        }
    }
}

/**
 * Kernel for calcium-dependent channel modulation
 * Updates channel properties based on local calcium concentrations
 */
__global__ void calciumDependentModulation(
    GPUNeuronState* neurons,
    float dt,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];

    if (neuron.active == 0) return;

    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;

        float ca_conc = neuron.ca_conc[c];

        // ========================================
        // CALCIUM-DEPENDENT POTASSIUM CHANNELS
        // ========================================
        // Update KCa channel activation based on local calcium
        KCaChannel kca_channel = INIT_KCA_CHANNEL(neuron.compartment_types[c]);
        float kca_inf = kca_channel.calciumDependentActivation(ca_conc);

        // Smooth transition to new activation level
        float kca_current = neuron.channels.kca_m[c];
        float dkca_dt = (kca_inf - kca_current) / kca_channel.tau_act;
        neuron.channels.kca_m[c] = kca_current + dkca_dt * dt;

        // Clamp to valid range
        neuron.channels.kca_m[c] = fmaxf(0.0f, fminf(1.0f, neuron.channels.kca_m[c]));

        // ========================================
        // CALCIUM-DEPENDENT NMDA MODULATION
        // ========================================
        // High calcium reduces NMDA effectiveness (calcium-dependent inactivation)
        float ca_inactivation = 1.0f / (1.0f + (ca_conc / 0.005f));  // Half-inactivation at 5 μM

        // Modulate NMDA conductance (not changing the state, just effectiveness)
        // This will be applied in the current calculation
        // Store modulation factor for use in neuron update kernel
        // (We could add a modulation field to the neuron structure)

        // ========================================
        // CALCIUM-DEPENDENT PLASTICITY THRESHOLDS
        // ========================================
        // High calcium lowers plasticity threshold (facilitates LTP)
        float ca_plasticity_factor = ca_conc / (ca_conc + 0.001f);  // Michaelis-Menten

        // Update plasticity threshold (lower values = easier plasticity)
        float base_threshold = 0.5f;
        neuron.plasticity_threshold = base_threshold * (1.0f - 0.5f * ca_plasticity_factor);

        // ========================================
        // CALCIUM-DEPENDENT SPIKE THRESHOLD
        // ========================================
        // Calcium affects excitability through various mechanisms
        if (c == 0) {  // Soma only
            float ca_excitability_factor = 1.0f + 0.2f * (ca_conc - RESTING_CA_CONCENTRATION) / RESTING_CA_CONCENTRATION;
            neuron.spike_threshold_modulated = neuron.spike_threshold * ca_excitability_factor;

            // Clamp threshold to reasonable range
            neuron.spike_threshold_modulated = fmaxf(-40.0f, fminf(10.0f, neuron.spike_threshold_modulated));
        }

        // ========================================
        // CALCIUM-DEPENDENT HOMEOSTASIS
        // ========================================
        // Calcium influences homeostatic scaling
        float ca_homeostasis_signal = (ca_conc - RESTING_CA_CONCENTRATION) / RESTING_CA_CONCENTRATION;

        // Update neuromodulation scaling based on calcium
        neuron.neuromod_excitability += 0.001f * dt * (-ca_homeostasis_signal);
        neuron.neuromod_excitability = fmaxf(0.5f, fminf(2.0f, neuron.neuromod_excitability));
    }
}

/**
 * Kernel for compartmental voltage coupling
 * Updates voltage coupling between compartments based on calcium and other factors
 */
__global__ void updateCompartmentalCoupling(
    GPUNeuronState* neurons,
    float dt,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];

    if (neuron.active == 0) return;

    for (int c = 1; c < neuron.compartment_count; c++) {  // Start from 1 (skip soma)
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;

        int parent = neuron.parent_compartment[c];
        if (parent < 0 || parent >= neuron.compartment_count) continue;

        // ========================================
        // CALCIUM-DEPENDENT COUPLING
        // ========================================
        float ca_child = neuron.ca_conc[c];
        float ca_parent = neuron.ca_conc[parent];
        float avg_ca = (ca_child + ca_parent) * 0.5f;

        // Calcium affects coupling strength
        float ca_coupling_factor = 1.0f + 0.5f * (avg_ca - RESTING_CA_CONCENTRATION) / RESTING_CA_CONCENTRATION;
        ca_coupling_factor = fmaxf(0.5f, fminf(2.0f, ca_coupling_factor));

        // ========================================
        // ACTIVITY-DEPENDENT COUPLING
        // ========================================
        float parent_voltage = (parent == 0) ? neuron.voltage : neuron.voltages[parent];
        float child_voltage = neuron.voltages[c];
        float voltage_diff = fabs(parent_voltage - child_voltage);

        // Large voltage differences increase coupling (gap junction voltage sensitivity)
        float voltage_coupling_factor = 1.0f + 0.1f * voltage_diff / 50.0f;  // Normalize by 50mV
        voltage_coupling_factor = fmaxf(0.8f, fminf(1.5f, voltage_coupling_factor));

        // ========================================
        // UPDATE COUPLING CONDUCTANCE
        // ========================================
        float base_coupling = 0.1f;  // Base coupling conductance (nS)

        // Compartment-type specific coupling
        if (neuron.compartment_types[c] == COMPARTMENT_SPINE) {
            base_coupling *= 0.5f;  // Weaker coupling for spines
        } else if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            base_coupling *= 1.2f;  // Stronger coupling for apical dendrites
        }

        // Apply modulation factors
        float new_coupling = base_coupling * ca_coupling_factor * voltage_coupling_factor;

        // Smooth transition
        float coupling_tau = 10.0f;  // Time constant for coupling changes (ms)
        float dcoupling_dt = (new_coupling - neuron.coupling_conductance[c]) / coupling_tau;
        neuron.coupling_conductance[c] += dcoupling_dt * dt;

        // Enforce bounds
        neuron.coupling_conductance[c] = fmaxf(0.01f, fminf(1.0f, neuron.coupling_conductance[c]));
    }
}

/**
 * Host function to launch calcium diffusion and related processes
 */
void launchCalciumDynamics(
    GPUNeuronState* d_neurons,
    float dt,
    int num_neurons
) {
    // Launch calcium diffusion kernel
    {
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);

        calciumDiffusionKernel<<<grid, block>>>(d_neurons, dt, num_neurons);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in calcium diffusion: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    // Launch calcium-dependent modulation kernel
    {
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);

        calciumDependentModulation<<<grid, block>>>(d_neurons, dt, num_neurons);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in calcium-dependent modulation: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    // Launch compartmental coupling update
    {
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);

        updateCompartmentalCoupling<<<grid, block>>>(d_neurons, dt, num_neurons);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in compartmental coupling update: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    cudaDeviceSynchronize();
}

/**
 * Kernel for calcium monitoring and statistics
 * Tracks calcium levels for debugging and analysis
 */
__global__ void monitorCalciumLevels(
    GPUNeuronState* neurons,
    float* ca_soma_avg,
    float* ca_dendrite_avg,
    float* ca_max_levels,
    int* ca_overflow_count,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[idx];

    if (neuron.active == 0) return;

    float soma_ca = 0.0f;
    float dendrite_ca_sum = 0.0f;
    float max_ca = 0.0f;
    int dendrite_count = 0;
    bool overflow_detected = false;

    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;

        float ca = neuron.ca_conc[c];
        max_ca = fmaxf(max_ca, ca);

        if (ca > MAX_CA_CONCENTRATION * 0.9f) {
            overflow_detected = true;
        }

        if (neuron.compartment_types[c] == COMPARTMENT_SOMA) {
            soma_ca = ca;
        } else {
            dendrite_ca_sum += ca;
            dendrite_count++;
        }
    }

    // Use atomic operations to accumulate statistics
    atomicAdd(ca_soma_avg, soma_ca);
    if (dendrite_count > 0) {
        atomicAdd(ca_dendrite_avg, dendrite_ca_sum / dendrite_count);
    }
    atomicAdd(&ca_max_levels[idx], max_ca);

    if (overflow_detected) {
        atomicAdd(ca_overflow_count, 1);
    }
}

/**
 * Host function to get calcium statistics
 */
void getCalciumStatistics(
    GPUNeuronState* d_neurons,
    int num_neurons,
    float* avg_soma_ca,
    float* avg_dendrite_ca,
    float* max_ca_level,
    int* overflow_neurons
) {
    float* d_ca_soma_avg;
    float* d_ca_dendrite_avg;
    float* d_ca_max_levels;
    int* d_ca_overflow_count;

    // Allocate device memory
    cudaMalloc(&d_ca_soma_avg, sizeof(float));
    cudaMalloc(&d_ca_dendrite_avg, sizeof(float));
    cudaMalloc(&d_ca_max_levels, num_neurons * sizeof(float));
    cudaMalloc(&d_ca_overflow_count, sizeof(int));

    // Initialize to zero
    cudaMemset(d_ca_soma_avg, 0, sizeof(float));
    cudaMemset(d_ca_dendrite_avg, 0, sizeof(float));
    cudaMemset(d_ca_max_levels, 0, num_neurons * sizeof(float));
    cudaMemset(d_ca_overflow_count, 0, sizeof(int));

    // Launch monitoring kernel
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);

    monitorCalciumLevels<<<grid, block>>>(
        d_neurons, d_ca_soma_avg, d_ca_dendrite_avg,
        d_ca_max_levels, d_ca_overflow_count, num_neurons
    );

    // Copy results back to host
    float h_soma_avg, h_dendrite_avg;
    int h_overflow_count;

    cudaMemcpy(&h_soma_avg, d_ca_soma_avg, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dendrite_avg, d_ca_dendrite_avg, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_overflow_count, d_ca_overflow_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate averages
    *avg_soma_ca = h_soma_avg / num_neurons;
    *avg_dendrite_ca = h_dendrite_avg / num_neurons;
    *overflow_neurons = h_overflow_count;

    // Find maximum calcium level
    float* h_max_levels = new float[num_neurons];
    cudaMemcpy(h_max_levels, d_ca_max_levels, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);

    *max_ca_level = 0.0f;
    for (int i = 0; i < num_neurons; i++) {
        *max_ca_level = fmaxf(*max_ca_level, h_max_levels[i]);
    }

    // Cleanup
    delete[] h_max_levels;
    cudaFree(d_ca_soma_avg);
    cudaFree(d_ca_dendrite_avg);
    cudaFree(d_ca_max_levels);
    cudaFree(d_ca_overflow_count);
}