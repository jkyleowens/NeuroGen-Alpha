#pragma once

#ifdef __CUDACC__
#include "GPUNeuralStructures.h"
#include "RandomStateInit.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DEFAULT_BLOCK_SIZE 256

// Neuron update kernel wrapper
void launchRK4NeuronUpdateKernel(GPUNeuronState* d_neurons, int num_neurons, float dt);

// Neuron spike detection wrapper
void launchUpdateNeuronSpikes(GPUNeuronState* d_neurons, int num_neurons, float threshold);

// Random state initialization
void launchRandomStateInit(curandState* d_states, int num_states, unsigned long seed);

// Spike event detection
void launchSpikeDetectionKernel(GPUNeuronState* neurons, GPUSpikeEvent* spikes,
                                float threshold, int* spike_count, int num_neurons, float current_time);

// Synapse input accumulation
void launchSynapseInputKernel(GPUSynapse* d_synapses, GPUNeuronState* d_neurons, int num_synapses);


#endif