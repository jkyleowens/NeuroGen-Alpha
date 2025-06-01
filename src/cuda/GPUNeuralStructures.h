#pragma once

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
