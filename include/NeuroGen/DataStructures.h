#ifndef NEUROGENALPHA_DATASTRUCTURES_H
#define NEUROGENALPHA_DATASTRUCTURES_H

#include "cuda/GPUNeuralStructures.h"

// Forward declarations for CPU-side structures
struct Neuron;
struct Synapse;
struct SpikeEvent;

// Conversion functions between CPU and GPU structures
GPUNeuronState convertToGPUNeuron(const Neuron& neuron);
GPUSynapse convertToGPUSynapse(const Synapse& synapse);
GPUSpikeEvent convertToGPUSpikeEvent(const SpikeEvent& event);

#endif // NEUROGENALPHA_DATASTRUCTURES_H