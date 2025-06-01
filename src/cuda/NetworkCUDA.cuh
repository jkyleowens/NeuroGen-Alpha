#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KernelLaunchWrappers.cuh"
#include "GPUNeuralStructures.h"

// Free function interface for main.cpp
void initializeNetwork();
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal);
void updateSynapticWeightsCUDA(float reward_signal);
void cleanupNetwork();

// CUDA kernel declarations
__global__ void injectInputCurrent(GPUNeuronState* input_neurons, float* input_data, 
                                  int input_size, float current_time);
__global__ void extractNeuralOutput(GPUNeuronState* output_neurons, float* output_buffer,
                                   int output_size, float current_time);
__global__ void applyRewardModulation(GPUNeuronState* neurons, int num_neurons, float reward);
__global__ void applyHomeostaticScaling(GPUSynapse* synapses, int num_synapses, float scale_factor);
__global__ void pruneSynapses(GPUSynapse* synapses, int num_synapses, float min_weight);