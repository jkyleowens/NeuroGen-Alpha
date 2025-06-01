#include "KernelLaunchWrappers.cuh"
#include "NeuronUpdateKernel.cuh"
#include "NeuronSpikingKernels.cuh"
#include "SynapseInputKernel.cuh"
#include "STDPKernel.cuh"
#include "RandomStateInit.cuh"
#include "GridBlockUtils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void launchUpdateNeuronVoltages(GPUNeuronState* neurons, float* I_leak, float* Cm, float dt, int N) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(N);
    updateNeuronVoltages<<<grid, block>>>(neurons, I_leak, Cm, dt, N);
}


void launchNeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(N);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, N);
}

void launchSynapseInputKernel(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_synapses);
    synapseInputKernel<<<grid, block>>>(d_synapses, d_neurons, num_synapses);
}

void launchRandomStateInit(curandState* d_states, int num_states, unsigned long seed) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_states);
    initializeRandomStates<<<grid, block>>>(d_states, num_states, seed);
}

void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(N);
    rk4NeuronUpdateKernel<<<grid, block>>>(neurons, dt, N);
}

void launchSpikeDetectionKernel(GPUNeuronState* neurons, GPUSpikeEvent* spikes, float threshold,
                                int* spike_count, int num_neurons, float current_time) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_neurons);
    detectSpikes<<<grid, block>>>(neurons, spikes, threshold, spike_count, num_neurons, current_time);
}

