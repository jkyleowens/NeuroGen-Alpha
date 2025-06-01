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

// Function declarations only - implementations are in STDPKernel.cu
extern "C" {
    // Simple STDP update for basic neural network (used by NetworkCUDA.cu)
    void launchSTDPUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                               int num_synapses, float A_plus, float A_minus,
                               float tau_plus, float tau_minus, float current_time,
                               float w_min, float w_max, float reward);
}

#endif // CORTICAL_STDP_KERNELS_CUH
