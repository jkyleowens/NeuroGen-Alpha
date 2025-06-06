#include "../../include/NeuroGen/cuda/STDPKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "../../include/NeuroGen/cuda/GridBlockUtils.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void stdpUpdateKernel(GPUSynapse* synapses, const GPUNeuronState* neurons,
                                int num_synapses, float A_plus, float A_minus,
                                float tau_plus, float tau_minus, float eligibility_decay,
                                float learning_rate, float current_time,
                                float min_weight, float max_weight, float reward_signal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Get spike times
    float t_pre = neurons[pre_idx].last_spike_time;
    float t_post = neurons[post_idx].last_spike_time;
    
    // Skip if no recent spikes
    if (t_pre < 0.0f || t_post < 0.0f) return;
    
    // Calculate time differences
    float dt_pre_post = t_post - t_pre;
    
    // STDP update -> eligibility trace
    float dw = 0.0f;

    if (dt_pre_post > 0.0f && dt_pre_post < 50.0f) {
        dw = A_plus * expf(-dt_pre_post / tau_plus);
    } else if (dt_pre_post < 0.0f && dt_pre_post > -50.0f) {
        dw = -A_minus * expf(dt_pre_post / tau_minus);
    }

    // Decay eligibility trace based on time since last update
    float decay = expf(-(current_time - synapse.last_active) / eligibility_decay);
    synapse.eligibility_trace *= decay;
    synapse.eligibility_trace += dw;

    // Reward-modulated weight update
    float delta_w = learning_rate * reward_signal * synapse.eligibility_trace;
    if (delta_w != 0.0f) {
        synapse.weight += delta_w;
        if (synapse.weight < min_weight) synapse.weight = min_weight;
        if (synapse.weight > max_weight) synapse.weight = max_weight;
        if (delta_w > 0.0f) synapse.last_potentiation = current_time;
    }

    synapse.last_active = current_time;
}

void launchSTDPUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                           int num_synapses, float A_plus, float A_minus,
                           float tau_plus, float tau_minus, float eligibility_decay,
                           float learning_rate, float current_time,
                           float min_weight, float max_weight, float reward_signal) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_synapses);
    
    stdpUpdateKernel<<<grid, block>>>(d_synapses, d_neurons, num_synapses,
                                     A_plus, A_minus, tau_plus, tau_minus,
                                     eligibility_decay, learning_rate, current_time,
                                     min_weight, max_weight, reward_signal);
    
    cudaDeviceSynchronize();
}