#include <NeuroGen/cuda/EnhancedSTDPKernel.cuh>
#include <NeuroGen/LearningRuleConstants.h>
#include <math_constants.h> // For CUDART_PI_F

__global__ void enhancedSTDPKernel(GPUSynapse* synapses, const GPUNeuronState* neurons,
                                  float current_time, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];

    // --- 1. Get Spike Timing Information ---
    float t_pre = pre_neuron.last_spike_time;
    float t_post = post_neuron.last_spike_time;
    float dt_spike = t_post - t_pre;

    // --- 2. Calculate Calcium-Dependent Plasticity Window ---
    int compartment_idx = synapse.post_compartment;
    float local_calcium = post_neuron.ca_conc[compartment_idx];
    float calcium_factor = 0.0f;

    if (local_calcium > CA_THRESHOLD_LTP) {
        // High calcium levels promote Long-Term Potentiation (LTP)
        calcium_factor = (local_calcium - CA_THRESHOLD_LTP) / (MAX_CA_CONCENTRATION - CA_THRESHOLD_LTP);
        calcium_factor = fminf(1.0f, calcium_factor); // Potentiation factor [0, 1]
    } else if (local_calcium > CA_THRESHOLD_LTD) {
        // Intermediate calcium levels promote Long-Term Depression (LTD)
        calcium_factor = (local_calcium - CA_THRESHOLD_LTD) / (CA_THRESHOLD_LTP - CA_THRESHOLD_LTD);
        calcium_factor = -fminf(1.0f, 1.0f - calcium_factor); // Depression factor [-1, 0]
    }

    // --- 3. Calculate Spike-Timing-Dependent Component ---
    float timing_factor = 0.0f;
    if (fabsf(dt_spike) < 50.0f) { // 50ms STDP window
        if (dt_spike > 0) { // Pre-before-post (LTP)
            timing_factor = STDP_A_PLUS_EXC * expf(-dt_spike / STDP_TAU_PLUS);
        } else { // Post-before-pre (LTD)
            timing_factor = -STDP_A_MINUS_EXC * expf(dt_spike / STDP_TAU_MINUS);
        }
    }
    
    // --- 4. Combine Factors to Update FAST Eligibility Trace ---
    // The fast trace represents the immediate potential for change.
    // It's a combination of the timing-dependent potential and the calcium-driven state.
    if (fabsf(calcium_factor) > 0.01f && fabsf(timing_factor) > 0.0001f) {
        // Only update if both timing and calcium signals are present
        float dw = timing_factor * (1.0f + fabsf(calcium_factor)) * synapse.plasticity_modulation;
        atomicAdd(&synapse.eligibility_trace, dw);
    }
}