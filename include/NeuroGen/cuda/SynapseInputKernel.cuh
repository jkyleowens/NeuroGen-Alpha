#pragma once
#ifndef SYNAPSE_INPUT_KERNEL_H
#define SYNAPSE_INPUT_KERNEL_H

#include "GPUNeuralStructures.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Function declarations only - implementations are in .cu file
__global__ void applySynapticCurrents(const GPUSynapse* synapses, 
                                     int num_synapses, 
                                     float* input_currents, 
                                     const GPUNeuronState* neurons);

__global__ void synapseInputKernel(GPUSynapse* synapses, 
                                  GPUNeuronState* neurons, 
                                  int num_synapses);

#endif // SYNAPSE_INPUT_KERNEL_H