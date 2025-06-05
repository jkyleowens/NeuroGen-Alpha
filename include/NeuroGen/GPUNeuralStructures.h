#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include <cuda_runtime.h>

// Include column structures depending on build mode
#ifdef __CUDACC__
#include "CorticalColumn.h"
#else
#include "GPUStructuresFwd.h"
#endif

// Maximum number of compartments per neuron
#define MAX_COMPARTMENTS 5
// Maximum number of synaptic receptors per compartment
#define MAX_SYNAPTIC_RECEPTORS 4

/**
 * GPU-optimized structure for neuron state
 */
struct GPUNeuronState {
    float voltage;                     // Membrane potential
    float m;                           // Activation variable for sodium channel
    float h;                           // Inactivation variable for sodium channel
    float n;                           // Activation variable for potassium channel
    float I_ext;                       // External current input
    float x, y, z;                     // 3D position coordinates
    int type;                          // Neuron type (e.g., excitatory, inhibitory)
    bool spiked;                       // Whether the neuron has spiked in the current timestep
    int active;                        // Whether the neuron is active (1) or inactive (0)
    float last_spike_time;             // Time of the last spike
    int compartment_count;             // Number of compartments
    float voltages[MAX_COMPARTMENTS];  // Voltages for each compartment
    float I_leak[MAX_COMPARTMENTS];    // Leak currents for each compartment
    float Cm[MAX_COMPARTMENTS];        // Membrane capacitances for each compartment
    float receptor_conductances[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS]; // Synaptic receptor conductances
};

/**
 * GPU-optimized structure for synapses
 */
struct GPUSynapse {
    int pre_neuron;            // ID of the presynaptic neuron (legacy field)
    int post_neuron;           // ID of the postsynaptic neuron (legacy field)
    int pre_neuron_idx;        // Index of the presynaptic neuron
    int post_neuron_idx;       // Index of the postsynaptic neuron
    float weight;              // Synaptic weight
    float delay;               // Synaptic delay in milliseconds
    float last_active;         // Time of last activation
    int type;                  // Synapse type
    int active;                // Whether the synapse is active (1) or inactive (0)
    float last_pre_spike_time; // Time of last presynaptic spike
    float activity_metric;     // Metric of recent activity
    float last_potentiation;   // Time of last potentiation
    int post_compartment;      // Target compartment on postsynaptic neuron
    int receptor_index;        // Target receptor type
};

/**
 * GPU-optimized structure for spike events
 */
struct GPUSpikeEvent {
    int neuron_idx;    // Index of the neuron that spiked
    float time;        // Time of the spike
    float amplitude;   // Amplitude of the spike
};

#endif // GPU_NEURAL_STRUCTURES_H