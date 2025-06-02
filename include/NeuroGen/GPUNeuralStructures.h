// include/NeuroGen/GPUNeuralStructures.h
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* ------------------------------------------------------------------------- */
/*  Core POD structs shared by all CUDA kernels                               */
/* ------------------------------------------------------------------------- */

struct GPUNeuronState {
    float voltage          = -65.0f;   // membrane potential  (mV)
    bool  spiked           = false;    // flag set this step
    float last_spike_time  = -1.0f;    // (ms)

    /* Hodgkin–Huxley gating variables */
    float m = 0.05f;
    float h = 0.60f;
    float n = 0.32f;

    /* Optional multi-compartment support (max 4) */
    int   compartment_count = 1;
    float voltages[4]       = { -65.0f };
    float I_leak[4]         = { 0.0f };
    float Cm[4]             = { 1.0f };
};

struct GPUSynapse {
    int   pre_neuron_idx;
    int   post_neuron_idx;
    float weight;
    float delay;
    float last_pre_spike_time;
    float activity_metric;           // for STDP / pruning
};

struct GPUSpikeEvent {
    int   neuron_index;
    float spike_time;
};

struct GPUCorticalColumn {
    // range of neurons belonging to this column  [begin, end)
    int neuron_start;
    int neuron_end;

    // range of synapses whose *post* neuron lives in this column
    int synapse_start;
    int synapse_end;

    // fast access to working buffers (device ptrs live in NetworkCUDA.cu)
    float* d_local_dopamine;     // neuromodulator buffer (optional)
    unsigned int* d_local_rng_state; // rng for structural plasticity

    int size() const { return neuron_end - neuron_start; }
};