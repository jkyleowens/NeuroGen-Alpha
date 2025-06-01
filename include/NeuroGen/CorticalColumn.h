#pragma once
#include "GPUNeuralStructures.h"     // GPUNeuronState, GPUSynapse
#include <vector>
#include <cstdint>  // for uint32_t
#include <cuda_runtime.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#define HOST_DEVICE
#else
#define HOST_DEVICE __host__ __device__
#endif

/**
 *  A light-weight container representing one biological cortical column.
 *  Keeps GPU-resident flat ranges instead of pointers to ease bulk copies.
 */
struct GPUCorticalColumn {
    // range of neurons belonging to this column  [begin, end)
    int neuron_start;
    int neuron_end;

    // range of synapses whose *post* neuron lives in this column
    int synapse_start;
    int synapse_end;

    // fast access to working buffers (device ptrs live in NetworkCUDA.cu)
    float* d_local_dopamine;     // neuromodulator buffer (optional)
    uint32_t* d_local_rng_state; // rng for structural plasticity

    HOST_DEVICE
    int size() const { return neuron_end - neuron_start; }
};
