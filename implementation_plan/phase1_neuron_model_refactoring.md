# Phase 1: Neuron Model Refactoring

## Overview

The current implementation has a basic structure for multi-compartment neurons, but it doesn't properly implement dendritic computation. This phase will enhance the neuron model to support realistic dendritic processing, which is crucial for the advanced learning mechanisms we'll implement in later phases.

## Current Implementation Analysis

From the code analysis, we found:

- `GPUNeuronState` has arrays for compartment voltages and receptor conductances, but they're not fully utilized
- The `rk4NeuronUpdateKernel` primarily updates the soma voltage using Hodgkin-Huxley dynamics
- Compartment-specific processing is minimal, with most synaptic inputs aggregated at the neuron level
- Dendritic computation and compartment-specific dynamics are not implemented

## Implementation Tasks

### 1. Enhance GPUNeuronState Structure

```cpp
// Current structure (simplified)
struct GPUNeuronState {
    float voltage;                     // Membrane potential
    float m, h, n;                     // HH variables
    int compartment_count;             // Number of compartments
    float voltages[MAX_COMPARTMENTS];  // Voltages for each compartment
    float receptor_conductances[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS];
};

// Enhanced structure
struct GPUNeuronState {
    // Soma properties
    float voltage;                     // Somatic membrane potential
    float m, h, n;                     // HH variables for soma
    
    // Compartment properties
    int compartment_count;             // Number of compartments
    int compartment_types[MAX_COMPARTMENTS]; // Type of each compartment (basal, apical, etc.)
    float voltages[MAX_COMPARTMENTS];  // Voltages for each compartment
    
    // Ion channel states for each compartment
    float m_comp[MAX_COMPARTMENTS];    // Na activation for each compartment
    float h_comp[MAX_COMPARTMENTS];    // Na inactivation for each compartment
    float n_comp[MAX_COMPARTMENTS];    // K activation for each compartment
    float ca_conc[MAX_COMPARTMENTS];   // Calcium concentration
    
    // Synaptic properties
    float receptor_conductances[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS];
    float receptor_states[MAX_COMPARTMENTS][MAX_SYNAPTIC_RECEPTORS]; // Additional state variables
    
    // Dendritic spike properties
    bool dendritic_spike[MAX_COMPARTMENTS];  // Whether a dendritic spike occurred
    float last_dendritic_spike[MAX_COMPARTMENTS]; // Time of last dendritic spike
    
    // Compartment connectivity
    int parent_compartment[MAX_COMPARTMENTS]; // Parent compartment index (-1 for soma)
    float coupling_conductance[MAX_COMPARTMENTS]; // Conductance to parent
};
```

### 2. Update Neuron Update Kernel

Modify `rk4NeuronUpdateKernel` to process compartments independently:

```cpp
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // 1. Update soma with Hodgkin-Huxley dynamics (existing code)
    // ...
    
    // 2. Update each dendritic compartment
    for (int c = 1; c < neuron.compartment_count; c++) {
        // Skip inactive compartments
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Extract compartment state variables
        float v = neuron.voltages[c];
        float m = neuron.m_comp[c];
        float h = neuron.h_comp[c];
        float n = neuron.n_comp[c];
        float ca = neuron.ca_conc[c];
        
        // Calculate ionic currents based on compartment type
        float I_Na, I_K, I_L, I_Ca, I_coupling;
        
        // Different dynamics based on compartment type
        if (neuron.compartment_types[c] == COMPARTMENT_BASAL) {
            // Basal dendrite dynamics
            // ...
        } else if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            // Apical dendrite dynamics with calcium channels
            // ...
        }
        
        // Calculate coupling current to parent compartment
        int parent = neuron.parent_compartment[c];
        if (parent >= 0) {
            I_coupling = neuron.coupling_conductance[c] * (neuron.voltages[parent] - v);
        } else {
            I_coupling = 0.0f;
        }
        
        // RK4 integration for this compartment
        // ...
        
        // Check for dendritic spike threshold
        if (v > DENDRITIC_SPIKE_THRESHOLD && !neuron.dendritic_spike[c]) {
            neuron.dendritic_spike[c] = true;
            neuron.last_dendritic_spike[c] = current_time;
            
            // Propagate effect to soma and parent compartments
            // ...
        }
        
        // Update compartment state
        neuron.voltages[c] = v;
        neuron.m_comp[c] = m;
        neuron.h_comp[c] = h;
        neuron.n_comp[c] = n;
        neuron.ca_conc[c] = ca;
    }
    
    // 3. Integrate dendritic inputs to soma
    // ...
}
```

### 3. Implement Dendritic Spike Generation

Add a new kernel for detecting and processing dendritic spikes:

```cpp
__global__ void dendriticSpikeKernel(GPUNeuronState* neurons, float current_time, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Process each compartment
    for (int c = 1; c < neuron.compartment_count; c++) {
        // Check if dendritic spike occurred
        if (neuron.dendritic_spike[c]) {
            // Reset if spike is old
            if (current_time - neuron.last_dendritic_spike[c] > DENDRITIC_SPIKE_DURATION) {
                neuron.dendritic_spike[c] = false;
            } else {
                // Propagate dendritic spike effect
                // For apical dendrites: calcium influx
                if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
                    neuron.ca_conc[c] += DENDRITIC_SPIKE_CA_INFLUX;
                }
                
                // Affect parent compartments
                int parent = neuron.parent_compartment[c];
                while (parent >= 0) {
                    // Depolarize parent compartment
                    neuron.voltages[parent] += DENDRITIC_SPIKE_PROPAGATION_STRENGTH;
                    parent = neuron.parent_compartment[parent];
                }
            }
        }
    }
}
```

### 4. Refactor Synaptic Input Processing

Modify the `synapseInputKernel` to route inputs to specific compartments:

```cpp
__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Check if presynaptic neuron spiked
    if (neurons[pre_idx].spiked) {
        // Record spike time for STDP
        synapse.last_pre_spike_time = neurons[pre_idx].last_spike_time;
        
        // Update activity metric
        synapse.activity_metric = synapse.activity_metric * 0.99f + 0.01f;
        
        // Apply synaptic input to specific compartment
        int compartment = synapse.post_compartment;
        int receptor = synapse.receptor_index;
        
        // Ensure indices are valid
        if (compartment >= 0 && compartment < MAX_COMPARTMENTS &&
            receptor >= 0 && receptor < MAX_SYNAPTIC_RECEPTORS) {
            
            // Add synaptic conductance with location-dependent scaling
            float location_factor = 1.0f;
            
            // Different processing based on compartment type
            if (compartment > 0) {
                int comp_type = neurons[post_idx].compartment_types[compartment];
                
                if (comp_type == COMPARTMENT_BASAL) {
                    // Basal dendrites may have different scaling
                    location_factor = BASAL_DENDRITE_SCALING;
                } else if (comp_type == COMPARTMENT_APICAL) {
                    // Apical dendrites may have different scaling
                    location_factor = APICAL_DENDRITE_SCALING;
                    
                    // NMDA receptors in apical dendrites can trigger calcium influx
                    if (receptor == RECEPTOR_NMDA && synapse.weight > NMDA_THRESHOLD) {
                        neurons[post_idx].ca_conc[compartment] += NMDA_CA_INFLUX;
                    }
                }
            }
            
            // Apply scaled weight to receptor conductance
            atomicAdd(&neurons[post_idx].receptor_conductances[compartment][receptor], 
                     synapse.weight * location_factor);
        }
    }
}
```

### 5. Create Neuron Initialization Functions

Add functions to initialize the enhanced neuron structure:

```cpp
__global__ void initializeNeuronCompartments(GPUNeuronState* neurons, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Initialize soma (compartment 0)
    neuron.voltages[0] = RESTING_POTENTIAL;
    neuron.m_comp[0] = 0.05f;
    neuron.h_comp[0] = 0.6f;
    neuron.n_comp[0] = 0.32f;
    neuron.ca_conc[0] = RESTING_CA_CONCENTRATION;
    neuron.compartment_types[0] = COMPARTMENT_SOMA;
    neuron.parent_compartment[0] = -1; // No parent for soma
    
    // Initialize dendritic compartments based on neuron type
    if (idx % 4 == 0) { // Example: every 4th neuron has a different structure
        // Complex neuron with both basal and apical dendrites
        neuron.compartment_count = 5;
        
        // Basal dendrites (compartments 1-2)
        for (int c = 1; c <= 2; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.compartment_types[c] = COMPARTMENT_BASAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = BASAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
        }
        
        // Apical dendrites (compartments 3-4)
        for (int c = 3; c <= 4; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.compartment_types[c] = COMPARTMENT_APICAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = APICAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
        }
    } else {
        // Simple neuron with just basal dendrites
        neuron.compartment_count = 3;
        
        // Basal dendrites (compartments 1-2)
        for (int c = 1; c <= 2; c++) {
            neuron.voltages[c] = RESTING_POTENTIAL;
            neuron.m_comp[c] = 0.05f;
            neuron.h_comp[c] = 0.6f;
            neuron.n_comp[c] = 0.32f;
            neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
            neuron.compartment_types[c] = COMPARTMENT_BASAL;
            neuron.parent_compartment[c] = 0; // Connected to soma
            neuron.coupling_conductance[c] = BASAL_COUPLING_CONDUCTANCE;
            neuron.dendritic_spike[c] = false;
        }
    }
    
    // Initialize receptor conductances
    for (int c = 0; c < neuron.compartment_count; c++) {
        for (int r = 0; r < MAX_SYNAPTIC_RECEPTORS; r++) {
            neuron.receptor_conductances[c][r] = 0.0f;
            neuron.receptor_states[c][r] = 0.0f;
        }
    }
}
```

### 6. Update Constants and Definitions

Create a new header file with constants for the enhanced neuron model:

```cpp
// NeuronModelConstants.h
#ifndef NEURON_MODEL_CONSTANTS_H
#define NEURON_MODEL_CONSTANTS_H

// Compartment types
#define COMPARTMENT_INACTIVE 0
#define COMPARTMENT_SOMA 1
#define COMPARTMENT_BASAL 2
#define COMPARTMENT_APICAL 3
#define COMPARTMENT_AXON 4

// Receptor types
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3

// Biophysical constants
#define RESTING_POTENTIAL -65.0f
#define RESTING_CA_CONCENTRATION 0.0001f
#define DENDRITIC_SPIKE_THRESHOLD -30.0f
#define DENDRITIC_SPIKE_DURATION 5.0f
#define DENDRITIC_SPIKE_CA_INFLUX 0.1f
#define DENDRITIC_SPIKE_PROPAGATION_STRENGTH 5.0f
#define NMDA_THRESHOLD 0.5f
#define NMDA_CA_INFLUX 0.05f

// Coupling conductances
#define BASAL_COUPLING_CONDUCTANCE 0.5f
#define APICAL_COUPLING_CONDUCTANCE 0.3f

// Synaptic scaling factors
#define BASAL_DENDRITE_SCALING 1.2f
#define APICAL_DENDRITE_SCALING 0.8f

#endif // NEURON_MODEL_CONSTANTS_H
```

## Integration Plan

1. Create the new header files and update the existing ones
2. Modify the neuron update kernels to implement compartment-specific processing
3. Update the synaptic input kernel to route inputs to specific compartments
4. Add the dendritic spike detection and processing kernel
5. Update the network initialization code to set up the enhanced neuron structure
6. Test the changes with a simple network to ensure proper functioning

## Expected Outcomes

- Neurons with realistic multi-compartment structure
- Compartment-specific processing of synaptic inputs
- Dendritic spike generation and propagation
- Foundation for implementing advanced learning rules in later phases

## Dependencies

- Phase 1 must be completed before moving on to Phase 2 (Ion Channel Dynamics)
- The enhanced neuron model will be used by all subsequent phases
