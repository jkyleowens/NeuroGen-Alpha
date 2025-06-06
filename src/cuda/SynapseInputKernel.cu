#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Enhanced synaptic input kernel with receptor-specific targeting
 * Processes incoming spikes and updates appropriate ion channel conductances
 */
__global__ void enhancedSynapticInputKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    GPUSpikeEvent* spike_events,
    int num_synapses,
    int num_spike_events,
    float current_time,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    // Get postsynaptic neuron
    GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    if (post_neuron.active == 0) return;
    
    // Check if target compartment is valid
    int target_comp = synapse.post_compartment;
    if (target_comp >= post_neuron.compartment_count || 
        post_neuron.compartment_types[target_comp] == COMPARTMENT_INACTIVE) {
        return;
    }
    
    // ========================================
    // CHECK FOR PRESYNAPTIC SPIKE
    // ========================================
    bool spike_received = false;
    float spike_time = 0.0f;
    
    // Search for spikes from presynaptic neuron
    for (int s = 0; s < num_spike_events; s++) {
        if (spike_events[s].neuron_idx == synapse.pre_neuron_idx) {
            float spike_arrival_time = spike_events[s].time + synapse.delay;
            
            // Check if spike arrives during this timestep
            if (spike_arrival_time >= current_time && 
                spike_arrival_time < current_time + dt) {
                spike_received = true;
                spike_time = spike_arrival_time;
                break;
            }
        }
    }
    
    // ========================================
    // PROCESS VESICLE RELEASE
    // ========================================
    float neurotransmitter_release = 0.0f;
    
    if (spike_received) {
        // Update spike timing
        synapse.last_pre_spike_time = spike_time;
        synapse.last_active = current_time;
        
        // Probabilistic vesicle release
        bool release_occurs = (curand_uniform(&g_rng_state) < synapse.release_probability);
        // SynapseInputKernel.cu — Fixed implementation file
#include <NeuroGen/cuda/CudaCompatibility.h>
#include "../../include/NeuroGen/cuda/SynapseInputKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/GridBlockUtils.cuh"
#include <cuda_runtime.h>
#include "NeuronModelConstants.h"

__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Check if presynaptic neuron spiked
    if (neurons[pre_idx].spiked) {
        // Record spike time for STDP
        synapse.last_pre_spike_time = neurons[pre_idx].last_spike_time;
        
        // Update activity metric
        synapse.activity_metric = synapse.activity_metric * 0.99f + 0.01f;
        
        // Apply synaptic input to specific compartment
        int compartment = synapse.post_compartment;
        int receptor = synapse.receptor_index;
        
        // Ensure indices are valid
        if (compartment >= 0 && compartment < MAX_COMPARTMENTS &&
            receptor >= 0 && receptor < MAX_SYNAPTIC_RECEPTORS) {
            
            // Add synaptic conductance with location-dependent scaling
            float location_factor = 1.0f;
            
            // Different processing based on compartment type
            if (compartment > 0) {
                int comp_type = neurons[post_idx].compartment_types[compartment];
                
                if (comp_type == COMPARTMENT_BASAL) {
                    // Basal dendrites may have different scaling
                    location_factor = BASAL_DENDRITE_SCALING;
                } else if (comp_type == COMPARTMENT_APICAL) {
                    // Apical dendrites may have different scaling
                    location_factor = APICAL_DENDRITE_SCALING;
                    
                    // NMDA receptors in apical dendrites can trigger calcium influx
                    if (receptor == RECEPTOR_NMDA && synapse.weight > NMDA_THRESHOLD) {
                        neurons[post_idx].ca_conc[compartment] += NMDA_CA_INFLUX;
                    }
                }
            }
            
            // Apply scaled weight to receptor conductance
            float effective_weight = synapse.weight;
            
            // Apply neuromodulatory effects on synaptic transmission
            if (synapse.effective_weight != 0.0f) {
                effective_weight = synapse.effective_weight;
            }
            
            effective_weight *= location_factor;
            
            // Update receptor conductance
            atomicAdd(&neurons[post_idx].receptor_conductances[compartment][receptor], effective_weight);
            
            // Update ion channel states directly
            if (receptor == RECEPTOR_AMPA) {
                // AMPA channels: fast excitatory
                atomicAdd(&neurons[post_idx].channels.ampa_state[compartment], effective_weight);
                atomicAdd(&neurons[post_idx].channels.ampa_g[compartment], effective_weight);
            } 
            else if (receptor == RECEPTOR_NMDA) {
                // NMDA channels: slow excitatory with Mg2+ block
                atomicAdd(&neurons[post_idx].channels.nmda_state[compartment], effective_weight);
                atomicAdd(&neurons[post_idx].channels.nmda_g[compartment], effective_weight);
                
                // NMDA receptors trigger calcium influx
                if (effective_weight > 0.0f) {
                    // Calculate calcium influx based on voltage-dependent Mg2+ block
                    float v = neurons[post_idx].voltages[compartment];
                    float mg_block = 1.0f / (1.0f + 0.28f * expf(-0.062f * v));
                    float ca_influx = effective_weight * mg_block * NMDA_CA_FRACTION;
                    
                    // Apply calcium influx modulation from neuromodulators
                    float ca_mod = neurons[post_idx].ca_influx_modulation[compartment];
                    if (ca_mod > 0.0f) {
                        ca_influx *= ca_mod;
                    }
                    
                    // Add calcium
                    atomicAdd(&neurons[post_idx].ca_conc[compartment], ca_influx * NMDA_CA_INFLUX);
                }
            }
            else if (receptor == RECEPTOR_GABA_A) {
                // GABA-A channels: fast inhibitory
                atomicAdd(&neurons[post_idx].channels.gaba_a_state[compartment], -effective_weight);
                atomicAdd(&neurons[post_idx].channels.gaba_a_g[compartment], -effective_weight);
            }
            else if (receptor == RECEPTOR_GABA_B) {
                // GABA-B channels: slow inhibitory with G-protein coupling
                atomicAdd(&neurons[post_idx].channels.gaba_b_state[compartment], -effective_weight);
                atomicAdd(&neurons[post_idx].channels.gaba_b_g_protein[compartment], -effective_weight);
            }
            
            // Update receptor state variable for kinetics
            atomicAdd(&neurons[post_idx].receptor_states[compartment][receptor], 1.0f);
        }
    }
}

            neurotransmitter_release = (float)vesicles_released / 10.0f;  // Normalize to 0-1
            
            // Apply synaptic weight
            neurotransmitter_release *= synapse.effective_weight;
            
            // Apply neuromodulation
            neurotransmitter_release *= synapse.plasticity_modulation;
            
            // Update activity metrics
            synapse.activity_metric = synapse.activity_metric * 0.95f + 0.05f;
        }
    }
    
    // ========================================
    // UPDATE VESICLE POOL
    // ========================================
    // Vesicle recovery (independent of spike)
    if (synapse.vesicles_ready < synapse.vesicle_pool_size) {
        float recovery_rate = synapse.vesicle_recovery_rate * dt;
        int max_recoverable = synapse.vesicle_pool_size - synapse.vesicles_ready;
        float expected_recovery = recovery_rate * max_recoverable;
        
        // Stochastic recovery
        int vesicles_recovered = (int)(expected_recovery + 0.5f);  // Round to nearest
        synapse.vesicles_ready = min(synapse.vesicles_ready + vesicles_recovered, 
                                   synapse.vesicle_pool_size);
    }
    
    // ========================================
    // UPDATE TARGET RECEPTOR CONDUCTANCES
    // ========================================
    if (neurotransmitter_release > 0.0f) {
        int receptor_type = synapse.receptor_index;
        float scaled_input = neurotransmitter_release * synapse.receptor_weight_fraction;
        
        // Apply neuromodulator scaling
        switch (receptor_type) {
            case RECEPTOR_AMPA:
                scaled_input *= post_neuron.neuromod_ampa_scale;
                atomicAdd(&post_neuron.channels.ampa_state[target_comp], scaled_input);
                break;
                
            case RECEPTOR_NMDA:
                scaled_input *= post_neuron.neuromod_nmda_scale;
                atomicAdd(&post_neuron.channels.nmda_state[target_comp], scaled_input);
                break;
                
            case RECEPTOR_GABA_A:
                scaled_input *= post_neuron.neuromod_gaba_scale;
                atomicAdd(&post_neuron.channels.gaba_a_state[target_comp], scaled_input);
                break;
                
            case RECEPTOR_GABA_B:
                scaled_input *= post_neuron.neuromod_gaba_scale;
                atomicAdd(&post_neuron.channels.gaba_b_state[target_comp], scaled_input);
                break;
        }
        
        // Update legacy receptor conductances for backward compatibility
        atomicAdd(&post_neuron.receptor_conductances[target_comp][receptor_type], 
                  scaled_input * 0.1f);  // Scale down for legacy system
    }
    
    // ========================================
    // UPDATE PLASTICITY TRACES
    // ========================================
    if (spike_received) {
        // Update eligibility traces (for Phase 3)
        synapse.fast_trace = synapse.fast_trace * expf(-dt / 50.0f) + 1.0f;      // 50ms decay
        synapse.medium_trace = synapse.medium_trace * expf(-dt / 1000.0f) + 1.0f; // 1s decay
        synapse.slow_trace = synapse.slow_trace * expf(-dt / 60000.0f) + 1.0f;    // 1min decay
        
        // Update calcium trace (simplified model)
        synapse.calcium_trace = synapse.calcium_trace * expf(-dt / 200.0f) + 
                               post_neuron.ca_conc[target_comp];
        
        // Update recent activity measure
        synapse.recent_activity = synapse.recent_activity * 0.99f + 0.01f;
    } else {
        // Decay traces in absence of activity
        synapse.fast_trace *= expf(-dt / 50.0f);
        synapse.medium_trace *= expf(-dt / 1000.0f);
        synapse.slow_trace *= expf(-dt / 60000.0f);
        synapse.calcium_trace *= expf(-dt / 200.0f);
        synapse.recent_activity *= 0.999f;
    }
    
    // ========================================
    // UPDATE SYNAPTIC EFFICACY
    // ========================================
    // Simple short-term plasticity model
    if (spike_received) {
        // Short-term depression
        synapse.effective_weight *= 0.95f;  // 5% depression per spike
        
        // Update release probability (frequency-dependent)
        float time_since_last = current_time - synapse.last_pre_spike_time;
        if (time_since_last < 10.0f) {  // High frequency
            synapse.release_probability *= 0.9f;  // Decrease release probability
        }
    } else {
        // Recovery from short-term depression
        float recovery_rate = 0.02f * dt;  // 2% recovery per ms
        synapse.effective_weight += recovery_rate * (synapse.weight - synapse.effective_weight);
        
        // Recovery of release probability
        float prob_recovery_rate = 0.01f * dt;
        float base_prob = (synapse.type == 0) ? 0.4f : 0.6f;  // Base prob for exc/inh
        synapse.release_probability += prob_recovery_rate * (base_prob - synapse.release_probability);
    }
    
    // Clamp values
    synapse.effective_weight = fmaxf(synapse.min_weight, 
                                   fminf(synapse.max_weight, synapse.effective_weight));
    synapse.release_probability = fmaxf(0.1f, fminf(0.9f, synapse.release_probability));
}

/**
 * Kernel to compute postsynaptic current contributions
 * Separates current calculation from conductance updates for better performance
 */
__global__ void computePostsynapticCurrents(
    GPUNeuronState* neurons,
    float* excitatory_currents,
    float* inhibitory_currents,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    if (neuron.active == 0) return;
    
    float total_excitatory = 0.0f;
    float total_inhibitory = 0.0f;
    
    // Sum currents across all compartments
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        float v = (c == 0) ? neuron.voltage : neuron.voltages[c];
        
        // AMPA current
        float I_AMPA = neuron.channels.ampa_g[c] * (v - AMPA_REVERSAL);
        
        // NMDA current with Mg block
        float mg_block = 1.0f / (1.0f + (NMDA_MG_CONC / 3.57f) * expf(-0.062f * v));
        float I_NMDA = neuron.channels.nmda_g[c] * mg_block * (v - NMDA_REVERSAL);
        
        // GABA-A current
        float I_GABA_A = neuron.channels.gaba_a_g[c] * (v - GABA_A_REVERSAL);
        
        // GABA-B current
        float I_GABA_B = neuron.channels.gaba_b_g[c] * neuron.channels.gaba_b_g_protein[c] * 
                         (v - GABA_B_REVERSAL);
        
        total_excitatory += -(I_AMPA + I_NMDA);  // Negative because inward current
        total_inhibitory += -(I_GABA_A + I_GABA_B);  // Negative because outward current
    }
    
    // Store results
    excitatory_currents[idx] = total_excitatory;
    inhibitory_currents[idx] = total_inhibitory;
    
    // Update neuron-level metrics
    neuron.total_excitatory_input = total_excitatory;
    neuron.total_inhibitory_input = total_inhibitory;
}

/**
 * Kernel to update synaptic input rates and neuron activity metrics
 */
__global__ void updateSynapticActivityMetrics(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int* synapse_to_neuron_map,
    int* neuron_synapse_counts,
    int num_neurons,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    if (neuron.active == 0) return;
    
    // Calculate synaptic input rate
    float input_events = 0.0f;
    int synapse_count = neuron_synapse_counts[idx];
    int synapse_start = (idx == 0) ? 0 : neuron_synapse_counts[idx - 1];
    
    for (int s = 0; s < synapse_count; s++) {
        int synapse_idx = synapse_to_neuron_map[synapse_start + s];
        GPUSynapse& synapse = synapses[synapse_idx];
        
        if (synapse.activity_metric > 0.01f) {
            input_events += synapse.activity_metric;
        }
    }
    
    // Update input rate (Hz)
    neuron.synaptic_input_rate = neuron.synaptic_input_rate * 0.95f + 
                                0.05f * (input_events / (dt * 0.001f));
    
    // Update overall activity level
    float current_activity = fabs(neuron.total_excitatory_input) + 
                           fabs(neuron.total_inhibitory_input);
    neuron.activity_level = neuron.activity_level * 0.99f + 0.01f * current_activity;
}

/**
 * Host function to launch enhanced synaptic input processing
 */
void launchEnhancedSynapticInput(
    GPUSynapse* d_synapses,
    GPUNeuronState* d_neurons,
    GPUSpikeEvent* d_spike_events,
    float* d_excitatory_currents,
    float* d_inhibitory_currents,
    int* d_synapse_to_neuron_map,
    int* d_neuron_synapse_counts,
    int num_synapses,
    int num_neurons,
    int num_spike_events,
    float current_time,
    float dt
) {
    // Launch synaptic input kernel
    {
        dim3 block(256);
        dim3 grid((num_synapses + block.x - 1) / block.x);
        
        enhancedSynapticInputKernel<<<grid, block>>>(
            d_synapses, d_neurons, d_spike_events,
            num_synapses, num_spike_events, current_time, dt
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in enhanced synaptic input: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    
    // Launch postsynaptic current computation
    {
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);
        
        computePostsynapticCurrents<<<grid, block>>>(
            d_neurons, d_excitatory_currents, d_inhibitory_currents, num_neurons
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in postsynaptic current computation: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    
    // Launch activity metrics update
    {
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);
        
        updateSynapticActivityMetrics<<<grid, block>>>(
            d_neurons, d_synapses, d_synapse_to_neuron_map, d_neuron_synapse_counts,
            num_neurons, dt
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in activity metrics update: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    
    cudaDeviceSynchronize();
}

/**
 * Kernel for background synaptic noise injection
 * Provides realistic background activity to maintain network dynamics
 */
__global__ void injectBackgroundNoise(
    GPUNeuronState* neurons,
    curandState* rng_states,
    float noise_rate,
    float dt,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    if (neuron.active == 0) return;
    
    curandState local_state = rng_states[idx];
    
    // Calculate noise probability for this timestep
    float noise_prob = noise_rate * dt * 0.001f;  // Convert Hz to probability
    
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Excitatory noise (AMPA)
        if (curand_uniform(&local_state) < noise_prob) {
            float noise_strength = 0.1f + 0.1f * curand_uniform(&local_state);  // 0.1-0.2
            atomicAdd(&neuron.channels.ampa_state[c], noise_strength);
        }
        
        // Inhibitory noise (GABA-A) - lower rate
        if (curand_uniform(&local_state) < noise_prob * 0.3f) {
            float noise_strength = 0.05f + 0.05f * curand_uniform(&local_state);  // 0.05-0.1
            atomicAdd(&neuron.channels.gaba_a_state[c], noise_strength);
        }
    }
    
    rng_states[idx] = local_state;
}

/**
 * Host function to inject background synaptic noise
 */
void injectSynapticNoise(
    GPUNeuronState* d_neurons,
    curandState* d_rng_states,
    float noise_rate_hz,
    float dt,
    int num_neurons
) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    injectBackgroundNoise<<<grid, block>>>(
        d_neurons, d_rng_states, noise_rate_hz, dt, num_neurons
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in background noise injection: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cudaDeviceSynchronize();
}