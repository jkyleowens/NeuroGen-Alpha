// NeuronSpikingKernels.cu – implementations
#include "../../include/NeuroGen/cuda/NeuronSpikingKernels.cuh"
#include <device_launch_parameters.h>

/* ------------------------------------------------------------------------- */
/* 1. Count how many neurons spiked                                           */
/* ------------------------------------------------------------------------- */
__global__ void countSpikesKernel(const GPUNeuronState* neurons,
                                  int                  num_neurons,
                                  int*                 spike_count)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_neurons) return;

    if (neurons[idx].spiked)
        atomicAdd(spike_count, 1);
}

/* ------------------------------------------------------------------------- */
/* 2. Update per-neuron “spiked” flag from voltage                            */
/* ------------------------------------------------------------------------- */
__global__ void updateNeuronSpikes(GPUNeuronState* neurons,
                                   int             num_neurons,
                                   float           threshold)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_neurons) return;

    neurons[idx].spiked = (neurons[idx].voltage >= threshold);
}

/* ------------------------------------------------------------------------- */
/* 3. Detect spikes and store events                                          */
/* ------------------------------------------------------------------------- */
__global__ void detectSpikes(const GPUNeuronState* neurons,
                             GPUSpikeEvent*        spike_buffer,
                             float                 threshold,
                             int*                  spike_count,
                             int                   num_neurons,
                             float                 current_time)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_neurons) return;

    if (neurons[idx].voltage >= threshold)
    {
        int write_idx = atomicAdd(spike_count, 1);
        spike_buffer[write_idx].neuron_index = idx;
        spike_buffer[write_idx].spike_time   = current_time;
    }
}
