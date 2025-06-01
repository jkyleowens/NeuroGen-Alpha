#pragma once
#ifndef CORTICAL_STDP_KERNELS_CUH
#define CORTICAL_STDP_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUNeuralStructures.h"

// Forward declarations only - no function definitions in header!

// Cortical-specific plasticity parameters
struct CorticalPlasticityParams {
    // Layer-specific STDP parameters
    float layer_A_plus[6];     // Potentiation strength per layer
    float layer_A_minus[6];    // Depression strength per layer
    float layer_tau_plus[6];   // Potentiation time constant per layer
    float layer_tau_minus[6];  // Depression time constant per layer
    
    // Cell-type specific modulation
    float pyramidal_plasticity_gain;
    float interneuron_plasticity_gain;
    float stellate_plasticity_gain;
    
    // Homeostatic parameters
    float homeostatic_strength;
    float target_firing_rate;
    float metaplasticity_rate;
    
    // Neuromodulation sensitivity
    float dopamine_stdp_modulation;
    float acetylcholine_stdp_modulation;
    float norepinephrine_stdp_modulation;
};
    0.01f, // homeostatic_strength
    5.0f,  // target_firing_rate (Hz)
    0.001f, // metaplasticity_rate
    
    // Neuromodulation
    0.2f,  // dopamine_stdp_modulation
    0.1f,  // acetylcholine_stdp_modulation
    0.15f  // norepinephrine_stdp_modulation
};

// Get cell-type specific plasticity gain
__device__ float getCellTypePlasticityGain(NeuronType cell_type) {
    switch (cell_type) {
        case NeuronType::PYRAMIDAL_L23:
        case NeuronType::PYRAMIDAL_L5:
        case NeuronType::PYRAMIDAL_L6:
            return c_plasticity_params.pyramidal_plasticity_gain;
            
        case NeuronType::STELLATE_L4:
            return c_plasticity_params.stellate_plasticity_gain;
            
        case NeuronType::INTERNEURON_FS:
        case NeuronType::INTERNEURON_RS:
        case NeuronType::INTERNEURON_IS:
            return c_plasticity_params.interneuron_plasticity_gain;
            
        default:
            return 1.0f;
    }
}

// Layer-specific plasticity parameters
__device__ void getLayerSTDPParams(CorticalLayer layer, float& A_plus, float& A_minus, 
                                 float& tau_plus, float& tau_minus) {
    int layer_idx = (int)layer - 1; // Convert to 0-based index
    layer_idx = max(0, min(5, layer_idx)); // Clamp to valid range
    
    A_plus = c_plasticity_params.layer_A_plus[layer_idx];
    A_minus = c_plasticity_params.layer_A_minus[layer_idx];
    tau_plus = c_plasticity_params.layer_tau_plus[layer_idx];
    tau_minus = c_plasticity_params.layer_tau_minus[layer_idx];
}

// Neuromodulation-dependent weight change
__device__ float computeNeuromodulatedDeltaW(float base_delta_w, 
                                           float dopamine_level,
                                           float acetylcholine_level,
                                           float norepinephrine_level,
                                           bool is_reward_context) {
    
    float modulation = 1.0f;
    
    // Dopamine enhances plasticity in reward contexts
    if (is_reward_context) {
        modulation += c_plasticity_params.dopamine_stdp_modulation * dopamine_level;
    }
    
    // Acetylcholine enhances attention-related plasticity
    modulation += c_plasticity_params.acetylcholine_stdp_modulation * acetylcholine_level;
    
    // Norepinephrine modulates stress/arousal-related plasticity
    modulation += c_plasticity_params.norepinephrine_stdp_modulation * norepinephrine_level;
    
    return base_delta_w * modulation;
}

// Homeostatic scaling function
__device__ float computeHomeostaticScaling(float current_firing_rate, 
                                         float target_firing_rate,
                                         float homeostatic_strength) {
    float rate_error = current_firing_rate - target_firing_rate;
    float scaling_factor = 1.0f - homeostatic_strength * rate_error / target_firing_rate;
    return fmaxf(0.1f, fminf(2.0f, scaling_factor)); // Clamp between 0.1 and 2.0
}

// Metaplasticity: plasticity of plasticity based on recent activity
__device__ float computeMetaplasticityThreshold(float activity_history, 
                                              float base_threshold,
                                              float metaplasticity_rate) {
    // Higher recent activity increases plasticity threshold (BCM-like rule)
    float threshold_shift = metaplasticity_rate * (activity_history - base_threshold);
    return base_threshold + threshold_shift;
}

// Connection-type specific plasticity rules
__device__ float getConnectionTypeModulation(CorticalLayer pre_layer, CorticalLayer post_layer,
                                           NeuronType pre_type, NeuronType post_type) {
    
    // Excitatory-to-excitatory connections (E→E)
    if ((pre_type == NeuronType::PYRAMIDAL_L23 || pre_type == NeuronType::PYRAMIDAL_L5 || 
         pre_type == NeuronType::PYRAMIDAL_L6 || pre_type == NeuronType::STELLATE_L4) &&
        (post_type == NeuronType::PYRAMIDAL_L23 || post_type == NeuronType::PYRAMIDAL_L5 || 
         post_type == NeuronType::PYRAMIDAL_L6 || post_type == NeuronType::STELLATE_L4)) {
        
        // Layer-specific E→E modulation
        if (pre_layer == CorticalLayer::LAYER_4 && post_layer == CorticalLayer::LAYER_2) {
            return 1.2f; // Enhanced bottom-up plasticity
        } else if (pre_layer == CorticalLayer::LAYER_6 && post_layer == CorticalLayer::LAYER_4) {
            return 0.8f; // Reduced top-down plasticity
        }
        return 1.0f;
    }
    
    // Excitatory-to-inhibitory connections (E→I)
    else if ((pre_type == NeuronType::PYRAMIDAL_L23 || pre_type == NeuronType::PYRAMIDAL_L5 || 
              pre_type == NeuronType::PYRAMIDAL_L6 || pre_type == NeuronType::STELLATE_L4) &&
             (post_type == NeuronType::INTERNEURON_FS || post_type == NeuronType::INTERNEURON_RS || 
              post_type == NeuronType::INTERNEURON_IS)) {
        return 0.8f; // Moderate E→I plasticity
    }
    
    // Inhibitory-to-excitatory connections (I→E)
    else if ((pre_type == NeuronType::INTERNEURON_FS || pre_type == NeuronType::INTERNEURON_RS || 
              pre_type == NeuronType::INTERNEURON_IS) &&
             (post_type == NeuronType::PYRAMIDAL_L23 || post_type == NeuronType::PYRAMIDAL_L5 || 
              post_type == NeuronType::PYRAMIDAL_L6 || post_type == NeuronType::STELLATE_L4)) {
        return 0.6f; // Reduced I→E plasticity
    }
    
    // Inhibitory-to-inhibitory connections (I→I)
    else {
        return 0.5f; // Limited I→I plasticity
    }
}

// Enhanced STDP kernel with cortical specificity
__global__ void corticalSTDPUpdateKernel(GPUCorticalSynapse* synapses,
                                        const GPUCorticalNeuron* neurons,
                                        const GPUCorticalColumn* columns,
                                        int num_synapses,
                                        float current_time,
                                        float global_reward,
                                        bool homeostasis_enabled) {
    
    int syn_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (syn_idx >= num_synapses) return;
    
    GPUCorticalSynapse& synapse = synapses[syn_idx];
    const GPUCorticalNeuron& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUCorticalNeuron& post_neuron = neurons[synapse.post_neuron_idx];
    const GPUCorticalColumn& pre_column = columns[pre_neuron.column_id];
    const GPUCorticalColumn& post_column = columns[post_neuron.column_id];
    
    // Only update if both neurons have spiked recently
    if (pre_neuron.last_spike_time < 0 || post_neuron.last_spike_time < 0) return;
    
    float dt = post_neuron.last_spike_time - pre_neuron.last_spike_time;
    float abs_dt = fabsf(dt);
    
    // Skip if spikes are too far apart (>100ms)
    if (abs_dt > 100.0f) return;
    
    // Get layer-specific STDP parameters
    float A_plus, A_minus, tau_plus, tau_minus;
    getLayerSTDPParams(post_neuron.layer, A_plus, A_minus, tau_plus, tau_minus);
    
    // Cell-type specific modulation
    float pre_gain = getCellTypePlasticityGain(pre_neuron.cell_type);
    float post_gain = getCellTypePlasticityGain(post_neuron.cell_type);
    float cell_type_modulation = sqrtf(pre_gain * post_gain);
    
    // Connection-type specific modulation
    float connection_modulation = getConnectionTypeModulation(
        pre_neuron.layer, post_neuron.layer,
        pre_neuron.cell_type, post_neuron.cell_type
    );
    
    // Calculate base STDP weight change
    float delta_w = 0.0f;
    if (dt > 0) {
        // Post before pre: potentiation
        delta_w = A_plus * expf(-dt / tau_plus);
    } else {
        // Pre before post: depression
        delta_w = -A_minus * expf(dt / tau_minus);
    }
    
    // Apply cell-type and connection-type modulation
    delta_w *= cell_type_modulation * connection_modulation;
    
    // Metaplasticity: adjust based on postsynaptic neuron's activity history
    float meta_threshold = computeMetaplasticityThreshold(
        post_neuron.activity_trace,
        post_neuron.plasticity_threshold,
        c_plasticity_params.metaplasticity_rate
    );
    
    // Only apply plasticity if above metaplasticity threshold
    if (post_neuron.activity_trace > meta_threshold) {
        
        // Neuromodulation (use average of pre and post columns)
        float avg_dopamine = 0.5f * (pre_column.dopamine_level + post_column.dopamine_level);
        float avg_acetylcholine = 0.5f * (pre_column.acetylcholine_level + post_column.acetylcholine_level);
        float avg_norepinephrine = 0.5f * (pre_column.norepinephrine_level + post_column.norepinephrine_level);
        
        bool reward_context = (global_reward > 0.0f);
        delta_w = computeNeuromodulatedDeltaW(delta_w, avg_dopamine, avg_acetylcholine, 
                                            avg_norepinephrine, reward_context);
        
        // Homeostatic scaling
        if (homeostasis_enabled) {
            float homeostatic_scaling = computeHomeostaticScaling(
                post_neuron.firing_rate_avg,
                c_plasticity_params.target_firing_rate,
                c_plasticity_params.homeostatic_strength
            );
            delta_w *= homeostatic_scaling;
        }
        
        // Apply weight change with bounds checking
        float new_weight = synapse.weight + delta_w;
        new_weight = fmaxf(synapse.min_weight, fminf(synapse.max_weight, new_weight));
        
        // Soft bounds to prevent saturation
        if (new_weight > 0.9f * synapse.max_weight) {
            new_weight = synapse.max_weight * (1.0f - expf(-10.0f * new_weight / synapse.max_weight));
        }
        
        synapse.weight = new_weight;
        
        // Update synapse activity metrics
        synapse.activity_metric += fabsf(delta_w);
        synapse.correlation_trace = 0.9f * synapse.correlation_trace + 0.1f * (dt > 0 ? 1.0f : -1.0f);
    }
    
    // Update spike timing records
    synapse.last_pre_spike_time = pre_neuron.last_spike_time;
    synapse.last_post_spike_time = post_neuron.last_spike_time;
}

// Short-term plasticity update (facilitation/depression)
__global__ void updateShortTermPlasticity(GPUCorticalSynapse* synapses,
                                         const GPUCorticalNeuron* neurons,
                                         int num_synapses,
                                         float dt) {
    
    int syn_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (syn_idx >= num_synapses) return;
    
    GPUCorticalSynapse& synapse = synapses[syn_idx];
    const GPUCorticalNeuron& pre_neuron = neurons[synapse.pre_neuron_idx];
    
    // Decay facilitation and recover depression
    synapse.u_current = synapse.u_0 + (synapse.u_current - synapse.u_0) * expf(-dt / synapse.tau_f);
    synapse.R_current = 1.0f + (synapse.R_current - 1.0f) * expf(-dt / synapse.tau_d);
    
    // If presynaptic neuron spiked, update short-term plasticity
    if (pre_neuron.spiked) {
        synapse.u_current += synapse.u_0 * (1.0f - synapse.u_current); // Facilitation
        synapse.R_current *= (1.0f - synapse.u_current); // Depression
        
        // Clamp values
        synapse.u_current = fminf(synapse.u_current, 1.0f);
        synapse.R_current = fmaxf(synapse.R_current, 0.0f);
    }
}

// Heterosynaptic plasticity: competition between synapses
__global__ void updateHeterosynapticPlasticity(GPUCorticalSynapse* synapses,
                                              const GPUCorticalNeuron* neurons,
                                              int num_synapses,
                                              float competition_strength) {
    
    int syn_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (syn_idx >= num_synapses) return;
    
    GPUCorticalSynapse& synapse = synapses[syn_idx];
    const GPUCorticalNeuron& post_neuron = neurons[synapse.post_neuron_idx];
    
    // Calculate total synaptic input to postsynaptic neuron
    float total_input = post_neuron.I_AMPA + post_neuron.I_NMDA;
    
    if (total_input > 0.0f) {
        // Normalize synaptic strength relative to total input
        float relative_strength = (synapse.weight * synapse.u_current * synapse.R_current) / total_input;
        
        // Heterosynaptic depression: weaker synapses get depressed
        if (relative_strength < 0.3f) { // Below threshold
            float depression = competition_strength * (0.3f - relative_strength);
            synapse.weight *= (1.0f - depression);
            synapse.weight = fmaxf(synapse.weight, synapse.min_weight);
        }
    }
}

// Global learning rate adaptation based on network state
__global__ void adaptGlobalLearningRate(GPUCorticalNetworkState* network_state,
                                       const GPUCorticalNeuron* neurons,
                                       int num_neurons,
                                       float target_activity) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate average network activity
        float total_activity = 0.0f;
        for (int i = 0; i < num_neurons; ++i) {
            total_activity += neurons[i].activity_trace;
        }
        float avg_activity = total_activity / num_neurons;
        
        // Adapt learning rate based on activity
        float activity_error = avg_activity - target_activity;
        float adaptation_rate = 0.001f;
        
        float new_learning_rate = network_state->global_learning_rate * 
                                 (1.0f - adaptation_rate * activity_error);
        
        network_state->global_learning_rate = fmaxf(0.001f, fminf(0.1f, new_learning_rate));
        network_state->average_firing_rate = avg_activity;
    }
}

// Wrapper functions for kernel launches (to be called from .cu file)
extern "C" {
    void launchCorticalSTDPUpdate(GPUCorticalSynapse* d_synapses,
                                 const GPUCorticalNeuron* d_neurons,
                                 const GPUCorticalColumn* d_columns,
                                 int num_synapses,
                                 float current_time,
                                 float global_reward,
                                 bool homeostasis_enabled);
    
    void launchShortTermPlasticityUpdate(GPUCorticalSynapse* d_synapses,
                                       const GPUCorticalNeuron* d_neurons,
                                       int num_synapses,
                                       float dt);
    
    void launchHeterosynapticPlasticityUpdate(GPUCorticalSynapse* d_synapses,
                                            const GPUCorticalNeuron* d_neurons,
                                            int num_synapses,
                                            float competition_strength);
    
    void launchGlobalLearningRateAdaptation(GPUCorticalNetworkState* d_network_state,
                                          const GPUCorticalNeuron* d_neurons,
                                          int num_neurons,
                                          float target_activity);
    
    // Simple STDP update for basic neural network (used by NetworkCUDA.cu)
    void launchSTDPUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                               int num_synapses, float A_plus, float A_minus,
                               float tau_plus, float tau_minus, float current_time,
                               float w_min, float w_max, float reward);
}

#endif // CORTICAL_STDP_KERNELS_CUH