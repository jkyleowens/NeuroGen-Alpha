// STDPKernel.cu - Implementation file
#include "STDPKernel.cuh"
#include "GPUNeuralStructures.h"
#include "GridBlockUtils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Replace missing RewardModulation.h with inline implementation
class NeuromodulatorState {
public:
    __host__ __device__ NeuromodulatorState() : reward_factor(1.0f) {}
    
    __host__ __device__ void apply_reward(float reward) {
        reward_factor = 1.0f + 0.1f * reward; // Simple scaling based on reward
    }
    
    __host__ __device__ float modulate_weight(float delta_w) {
        return delta_w * reward_factor;
    }
    
private:
    float reward_factor;
};

__global__ void stdpUpdateKernel(GPUSynapse* synapses, const GPUNeuronState* neurons,
                                 int num_synapses, float A_plus, float A_minus,
                                 float tau_plus, float tau_minus, float current_time,
                                 float w_min, float w_max, float reward) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_synapses) return;

    GPUSynapse& syn = synapses[i];
    float t_pre = neurons[syn.pre_neuron_idx].last_spike_time;
    float t_post = neurons[syn.post_neuron_idx].last_spike_time;
    float dt = t_post - t_pre;

    NeuromodulatorState nm;
    nm.apply_reward(reward);

    if (t_pre > 0 && t_post > 0) {
        float delta_w = 0.0f;
        if (dt > 0) delta_w = A_plus * expf(-dt / tau_plus);
        else if (dt < 0) delta_w = -A_minus * expf(dt / tau_minus);

        delta_w = nm.modulate_weight(delta_w);

        syn.weight += delta_w;
        syn.weight = fminf(fmaxf(syn.weight, w_min), w_max);
    }
}


void launchSTDPUpdateKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                            int num_synapses, float A_plus, float A_minus,
                            float tau_plus, float tau_minus, float current_time,
                            float w_min, float w_max, float reward) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_synapses);
    stdpUpdateKernel<<<grid, block>>>(
        d_synapses, d_neurons, num_synapses,
        A_plus, A_minus, tau_plus, tau_minus,
        current_time, w_min, w_max, reward
    );
}