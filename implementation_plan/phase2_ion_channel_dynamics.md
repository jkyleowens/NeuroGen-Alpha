# Phase 2: Ion Channel Dynamics Implementation

## Overview

The current implementation has placeholders for ion channels but doesn't fully implement their dynamics. This phase will expand the ion channel models to include NMDA, AMPA, GABA-A, GABA-B, and voltage-gated calcium channels, providing a more biologically realistic foundation for learning and computation.

## Current Implementation Analysis

From the code analysis, we found:

- Basic Hodgkin-Huxley dynamics are implemented for the soma
- The `rk4NeuronUpdateKernel` includes Na+, K+, and leak currents
- Receptor conductances are tracked but not fully utilized in the dynamics
- Calcium dynamics and other ion channels are missing

## Implementation Tasks

### 1. Expand Ion Channel Models

#### 1.1 Define Ion Channel Structures

Create a comprehensive set of ion channel models:

```cpp
// IonChannelModels.h
#ifndef ION_CHANNEL_MODELS_H
#define ION_CHANNEL_MODELS_H

#include <cuda_runtime.h>
#include "NeuronModelConstants.h"

// AMPA receptor model
struct AMPAChannel {
    float g_max;        // Maximum conductance
    float tau_rise;     // Rise time constant
    float tau_decay;    // Decay time constant
    float reversal;     // Reversal potential
    
    __device__ float computeCurrent(float v, float g) {
        return g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Dual exponential synapse model
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
    }
};

// NMDA receptor model with Mg2+ block
struct NMDAChannel {
    float g_max;        // Maximum conductance
    float tau_rise;     // Rise time constant
    float tau_decay;    // Decay time constant
    float reversal;     // Reversal potential
    float mg_conc;      // Magnesium concentration
    
    __device__ float computeMgBlock(float v) {
        // Voltage-dependent magnesium block
        return 1.0f / (1.0f + (mg_conc / 3.57f) * expf(-0.062f * v));
    }
    
    __device__ float computeCurrent(float v, float g) {
        float mg_block = computeMgBlock(v);
        return g * mg_block * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Dual exponential synapse model with slower kinetics than AMPA
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
    }
};

// GABA-A receptor model (fast inhibition)
struct GABAA_Channel {
    float g_max;        // Maximum conductance
    float tau_rise;     // Rise time constant
    float tau_decay;    // Decay time constant
    float reversal;     // Reversal potential (typically -70 mV)
    
    __device__ float computeCurrent(float v, float g) {
        return g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Fast inhibitory synapse dynamics
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
    }
};

// GABA-B receptor model (slow inhibition with G-protein coupling)
struct GABAB_Channel {
    float g_max;        // Maximum conductance
    float tau_rise;     // Rise time constant for G-protein activation
    float tau_decay;    // Decay time constant for G-protein deactivation
    float tau_k;        // Time constant for K+ channel activation
    float reversal;     // Reversal potential (typically -90 mV, K+ reversal)
    float hill_coef;    // Hill coefficient for G-protein activation
    
    __device__ float computeCurrent(float v, float g) {
        return g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float g_protein, float input, float dt) {
        // G-protein coupled receptor dynamics (metabotropic)
        float dg_protein_dt = -g_protein / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        // K+ channel activation via G-protein (with Hill function)
        float g_inf = powf(g_protein, hill_coef) / (powf(g_protein, hill_coef) + powf(0.5f, hill_coef));
        float dgdt = (g_inf - g) / tau_k;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
        g_protein += dg_protein_dt * dt;
    }
};

// Voltage-gated calcium channel
struct CaChannel {
    float g_max;        // Maximum conductance
    float reversal;     // Reversal potential (typically +50 mV)
    float v_half;       // Half-activation voltage
    float k;            // Slope factor
    float tau_act;      // Activation time constant
    
    __device__ float steadyStateActivation(float v) {
        return 1.0f / (1.0f + expf(-(v - v_half) / k));
    }
    
    __device__ float computeCurrent(float v, float m) {
        return g_max * m * m * (v - reversal);
    }
    
    __device__ void updateState(float& m, float v, float dt) {
        float m_inf = steadyStateActivation(v);
        float dmdt = (m_inf - m) / tau_act;
        m += dmdt * dt;
    }
};

// Calcium-dependent potassium channel (SK/BK type)
struct KCaChannel {
    float g_max;        // Maximum conductance
    float reversal;     // Reversal potential (K+ reversal, typically -90 mV)
    float ca_half;      // Half-activation calcium concentration
    float hill_coef;    // Hill coefficient for calcium dependence
    float tau_act;      // Activation time constant
    
    __device__ float steadyStateActivation(float ca_conc) {
        return powf(ca_conc, hill_coef) / (powf(ca_conc, hill_coef) + powf(ca_half, hill_coef));
    }
    
    __device__ float computeCurrent(float v, float m) {
        return g_max * m * (v - reversal);
    }
    
    __device__ void updateState(float& m, float ca_conc, float dt) {
        float m_inf = steadyStateActivation(ca_conc);
        float dmdt = (m_inf - m) / tau_act;
        m += dmdt * dt;
    }
};

// Hyperpolarization-activated cyclic nucleotide-gated (HCN) channel
struct HCNChannel {
    float g_max;        // Maximum conductance
    float reversal;     // Reversal potential (typically -30 mV)
    float v_half;       // Half-activation voltage
    float k;            // Slope factor
    float tau_min;      // Minimum time constant
    float tau_max;      // Maximum time constant
    float v_tau;        // Voltage at which time constant is midway between min and max
    float k_tau;        // Slope factor for time constant
    
    __device__ float steadyStateActivation(float v) {
        return 1.0f / (1.0f + expf((v - v_half) / k));  // Note: opposite slope from typical channels
    }
    
    __device__ float activationTimeConstant(float v) {
        return tau_min + (tau_max - tau_min) / (1.0f + expf(-(v - v_tau) / k_tau));
    }
    
    __device__ float computeCurrent(float v, float h) {
        return g_max * h * (v - reversal);
    }
    
    __device__ void updateState(float& h, float v, float dt) {
        float h_inf = steadyStateActivation(v);
        float tau_h = activationTimeConstant(v);
        float dhdt = (h_inf - h) / tau_h;
        h += dhdt * dt;
    }
};

#endif // ION_CHANNEL_MODELS_H
```

#### 1.2 Update GPUNeuronState Structure

Extend the neuron state structure to include the new ion channel states:

```cpp
struct GPUNeuronState {
    // Existing fields from Phase 1
    // ...
    
    // Ion channel states for each compartment
    struct {
        // Synaptic channels
        float ampa_g[MAX_COMPARTMENTS];        // AMPA conductance
        float ampa_state[MAX_COMPARTMENTS];    // AMPA state variable
        float nmda_g[MAX_COMPARTMENTS];        // NMDA conductance
        float nmda_state[MAX_COMPARTMENTS];    // NMDA state variable
        float gaba_a_g[MAX_COMPARTMENTS];      // GABA-A conductance
        float gaba_a_state[MAX_COMPARTMENTS];  // GABA-A state variable
        float gaba_b_g[MAX_COMPARTMENTS];      // GABA-B conductance
        float gaba_b_state[MAX_COMPARTMENTS];  // GABA-B state variable
        float gaba_b_g_protein[MAX_COMPARTMENTS]; // GABA-B G-protein level
        
        // Voltage-gated channels
        float ca_m[MAX_COMPARTMENTS];          // Ca channel activation
        float kca_m[MAX_COMPARTMENTS];         // KCa channel activation
        float hcn_h[MAX_COMPARTMENTS];         // HCN channel activation
    } channels;
    
    // Calcium dynamics
    float ca_conc[MAX_COMPARTMENTS];           // Calcium concentration
    float ca_buffer[MAX_COMPARTMENTS];         // Calcium buffer concentration
    float ca_pump_rate[MAX_COMPARTMENTS];      // Calcium extrusion rate
};
```

### 2. Implement Calcium Dynamics

Add a new kernel for calcium dynamics:

```cpp
__global__ void calciumDynamicsKernel(GPUNeuronState* neurons, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Process each compartment
    for (int c = 0; c < neuron.compartment_count; c++) {
        // Skip inactive compartments
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Current calcium concentration
        float ca = neuron.ca_conc[c];
        float ca_buffer = neuron.ca_buffer[c];
        
        // Calcium influx from voltage-gated calcium channels
        float I_Ca = 0.0f;
        if (neuron.compartment_types[c] == COMPARTMENT_SOMA || 
            neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            // Compute calcium current using the CaChannel model
            CaChannel ca_channel;
            ca_channel.g_max = 0.5f;  // Example value
            ca_channel.reversal = 50.0f;
            ca_channel.v_half = -20.0f;
            ca_channel.k = 9.0f;
            ca_channel.tau_act = 1.0f;
            
            I_Ca = ca_channel.computeCurrent(neuron.voltages[c], neuron.channels.ca_m[c]);
            ca_channel.updateState(neuron.channels.ca_m[c], neuron.voltages[c], dt);
        }
        
        // Calcium influx from NMDA receptors
        float I_NMDA_Ca = 0.0f;
        if (neuron.receptor_conductances[c][RECEPTOR_NMDA] > 0.0f) {
            // NMDA channels contribute to calcium influx
            NMDAChannel nmda_channel;
            nmda_channel.g_max = 1.0f;
            nmda_channel.reversal = 0.0f;
            nmda_channel.mg_conc = 1.0f;
            
            float nmda_g = neuron.receptor_conductances[c][RECEPTOR_NMDA];
            float mg_block = nmda_channel.computeMgBlock(neuron.voltages[c]);
            
            // Fraction of NMDA current carried by calcium
            float ca_fraction = 0.1f;
            I_NMDA_Ca = ca_fraction * nmda_g * mg_block * (neuron.voltages[c] - nmda_channel.reversal);
        }
        
        // Total calcium current
        float I_Ca_total = -(I_Ca + I_NMDA_Ca);  // Negative because inward current increases calcium
        
        // Convert current to concentration change
        // Assume a simple conversion factor based on compartment volume
        float volume_factor = (c == 0) ? 1.0f : 0.2f;  // Smaller for dendrites
        float ca_influx = I_Ca_total * 0.01f / volume_factor;
        
        // Calcium buffering (simple first-order kinetics)
        float buffer_forward_rate = 0.5f;  // Rate of calcium binding to buffer
        float buffer_reverse_rate = 0.1f;  // Rate of calcium unbinding from buffer
        float buffer_capacity = 10.0f;     // Total buffer capacity
        
        float free_buffer = buffer_capacity - ca_buffer;
        float buffering = buffer_forward_rate * ca * free_buffer - buffer_reverse_rate * ca_buffer;
        
        // Calcium extrusion (pump)
        float pump_rate = neuron.ca_pump_rate[c];
        float extrusion = pump_rate * ca / (ca + 0.0001f);  // Michaelis-Menten kinetics
        
        // Calcium diffusion between compartments
        float diffusion = 0.0f;
        if (c > 0) {  // Not soma
            int parent = neuron.parent_compartment[c];
            if (parent >= 0) {
                float ca_parent = neuron.ca_conc[parent];
                float diffusion_rate = 0.1f;  // Example value
                diffusion = diffusion_rate * (ca_parent - ca);
            }
        }
        
        // Update calcium concentration
        float dca_dt = ca_influx - buffering - extrusion + diffusion;
        float dca_buffer_dt = buffering;
        
        neuron.ca_conc[c] += dca_dt * dt;
        neuron.ca_buffer[c] += dca_buffer_dt * dt;
        
        // Ensure calcium stays in valid range
        if (neuron.ca_conc[c] < 0.0f) neuron.ca_conc[c] = 0.0f;
        if (neuron.ca_buffer[c] < 0.0f) neuron.ca_buffer[c] = 0.0f;
        if (neuron.ca_buffer[c] > buffer_capacity) neuron.ca_buffer[c] = buffer_capacity;
    }
}
```

### 3. Update Neuron Update Kernel

Modify the `rk4NeuronUpdateKernel` to incorporate the new ion channels:

```cpp
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons, float dt, float current_time, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Process each compartment
    for (int c = 0; c < neuron.compartment_count; c++) {
        // Skip inactive compartments
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Extract compartment state variables
        float v = neuron.voltages[c];
        float m, h, n;
        
        if (c == 0) {  // Soma uses the main HH variables
            m = neuron.m;
            h = neuron.h;
            n = neuron.n;
        } else {  // Dendrites use compartment-specific variables
            m = neuron.m_comp[c];
            h = neuron.h_comp[c];
            n = neuron.n_comp[c];
        }
        
        // Calculate basic Hodgkin-Huxley currents
        float I_Na = HH_G_NA * m*m*m * h * (v - HH_E_NA);
        float I_K = HH_G_K * n*n*n*n * (v - HH_E_K);
        float I_L = HH_G_L * (v - HH_E_L);
        
        // Calculate synaptic currents
        float I_syn = 0.0f;
        
        // AMPA current
        AMPAChannel ampa;
        ampa.g_max = 1.0f;
        ampa.tau_rise = 0.5f;
        ampa.tau_decay = 3.0f;
        ampa.reversal = 0.0f;
        
        float ampa_g = neuron.channels.ampa_g[c];
        float ampa_state = neuron.channels.ampa_state[c];
        float ampa_input = neuron.receptor_conductances[c][RECEPTOR_AMPA];
        
        I_syn += ampa.computeCurrent(v, ampa_g);
        ampa.updateState(ampa_g, ampa_state, ampa_input, dt);
        
        // NMDA current
        NMDAChannel nmda;
        nmda.g_max = 1.0f;
        nmda.tau_rise = 5.0f;
        nmda.tau_decay = 50.0f;
        nmda.reversal = 0.0f;
        nmda.mg_conc = 1.0f;
        
        float nmda_g = neuron.channels.nmda_g[c];
        float nmda_state = neuron.channels.nmda_state[c];
        float nmda_input = neuron.receptor_conductances[c][RECEPTOR_NMDA];
        
        I_syn += nmda.computeCurrent(v, nmda_g);
        nmda.updateState(nmda_g, nmda_state, nmda_input, dt);
        
        // GABA-A current
        GABAA_Channel gaba_a;
        gaba_a.g_max = 1.0f;
        gaba_a.tau_rise = 1.0f;
        gaba_a.tau_decay = 7.0f;
        gaba_a.reversal = -70.0f;
        
        float gaba_a_g = neuron.channels.gaba_a_g[c];
        float gaba_a_state = neuron.channels.gaba_a_state[c];
        float gaba_a_input = neuron.receptor_conductances[c][RECEPTOR_GABA_A];
        
        I_syn += gaba_a.computeCurrent(v, gaba_a_g);
        gaba_a.updateState(gaba_a_g, gaba_a_state, gaba_a_input, dt);
        
        // GABA-B current
        GABAB_Channel gaba_b;
        gaba_b.g_max = 0.5f;
        gaba_b.tau_rise = 50.0f;
        gaba_b.tau_decay = 100.0f;
        gaba_b.tau_k = 10.0f;
        gaba_b.reversal = -90.0f;
        gaba_b.hill_coef = 4.0f;
        
        float gaba_b_g = neuron.channels.gaba_b_g[c];
        float gaba_b_state = neuron.channels.gaba_b_state[c];
        float gaba_b_g_protein = neuron.channels.gaba_b_g_protein[c];
        float gaba_b_input = neuron.receptor_conductances[c][RECEPTOR_GABA_B];
        
        I_syn += gaba_b.computeCurrent(v, gaba_b_g);
        gaba_b.updateState(gaba_b_g, gaba_b_state, gaba_b_g_protein, gaba_b_input, dt);
        
        // Additional voltage-gated currents
        float I_additional = 0.0f;
        
        // Calcium current (only in soma and apical dendrites)
        if (neuron.compartment_types[c] == COMPARTMENT_SOMA || 
            neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            CaChannel ca_channel;
            ca_channel.g_max = 0.5f;
            ca_channel.reversal = 50.0f;
            ca_channel.v_half = -20.0f;
            ca_channel.k = 9.0f;
            ca_channel.tau_act = 1.0f;
            
            I_additional += ca_channel.computeCurrent(v, neuron.channels.ca_m[c]);
        }
        
        // Calcium-dependent potassium current
        KCaChannel kca_channel;
        kca_channel.g_max = 0.5f;
        kca_channel.reversal = -90.0f;
        kca_channel.ca_half = 0.001f;
        kca_channel.hill_coef = 4.0f;
        kca_channel.tau_act = 10.0f;
        
        I_additional += kca_channel.computeCurrent(v, neuron.channels.kca_m[c]);
        kca_channel.updateState(neuron.channels.kca_m[c], neuron.ca_conc[c], dt);
        
        // HCN channel (Ih, hyperpolarization-activated)
        HCNChannel hcn_channel;
        hcn_channel.g_max = 0.1f;
        hcn_channel.reversal = -30.0f;
        hcn_channel.v_half = -90.0f;
        hcn_channel.k = -10.0f;
        hcn_channel.tau_min = 50.0f;
        hcn_channel.tau_max = 500.0f;
        hcn_channel.v_tau = -75.0f;
        hcn_channel.k_tau = 15.0f;
        
        I_additional += hcn_channel.computeCurrent(v, neuron.channels.hcn_h[c]);
        hcn_channel.updateState(neuron.channels.hcn_h[c], v, dt);
        
        // Coupling current to parent compartment
        float I_coupling = 0.0f;
        if (c > 0) {  // Not soma
            int parent = neuron.parent_compartment[c];
            if (parent >= 0) {
                I_coupling = neuron.coupling_conductance[c] * (neuron.voltages[parent] - v);
            }
        }
        
        // Total current
        float I_total = -(I_Na + I_K + I_L + I_syn + I_additional) + I_coupling;
        
        // RK4 integration for voltage
        float k1_v = dt * I_total;
        // ... (similar to existing RK4 implementation)
        
        // Update compartment state
        if (c == 0) {  // Soma
            neuron.voltage = v;
            neuron.m = m;
            neuron.h = h;
            neuron.n = n;
        } else {  // Dendrites
            neuron.voltages[c] = v;
            neuron.m_comp[c] = m;
            neuron.h_comp[c] = h;
            neuron.n_comp[c] = n;
        }
        
        // Update channel states
        neuron.channels.ampa_g[c] = ampa_g;
        neuron.channels.ampa_state[c] = ampa_state;
        neuron.channels.nmda_g[c] = nmda_g;
        neuron.channels.nmda_state[c] = nmda_state;
        neuron.channels.gaba_a_g[c] = gaba_a_g;
        neuron.channels.gaba_a_state[c] = gaba_a_state;
        neuron.channels.gaba_b_g[c] = gaba_b_g;
        neuron.channels.gaba_b_state[c] = gaba_b_state;
        neuron.channels.gaba_b_g_protein[c] = gaba_b_g_protein;
        
        // Check for spike threshold crossing
        if (c == 0 && v > SPIKE_THRESHOLD && !neuron.spiked) {
            neuron.spiked = true;
            neuron.last_spike_time = current_time;
        }
    }
}
```

### 4. Update Synapse Structure

Enhance the synapse structure to include receptor-specific information:

```cpp
struct GPUSynapse {
    // Existing fields
    int pre_neuron_idx;
    int post_neuron_idx;
    float weight;
    float delay;
    float last_pre_spike_time;
    float activity_metric;
    int active;
    
    // New fields
    int post_compartment;     // Target compartment index
    int receptor_index;       // Target receptor type (AMPA, NMDA, etc.)
    float plasticity_rate;    // Learning rate for this synapse
    float eligibility_trace;  // Eligibility trace for reinforcement learning
};
```

### 5. Initialize Ion Channel Parameters

Create a function to initialize the ion channel parameters:

```cpp
__global__ void initializeIonChannels(GPUNeuronState* neurons, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Initialize calcium dynamics
    for (int c = 0; c < neuron.compartment_count; c++) {
        neuron.ca_conc[c] = RESTING_CA_CONCENTRATION;
        neuron.ca_buffer[c] = 0.0f;
        
        // Calcium pump rate varies by compartment type
        if (neuron.compartment_types[c] == COMPARTMENT_SOMA) {
            neuron.ca_pump_rate[c] = 0.2f;  // Faster in soma
        } else if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            neuron.ca_pump_rate[c] = 0.1f;  // Slower in apical dendrites
        } else {
            neuron.ca_pump_rate[c] = 0.15f; // Intermediate in basal dendrites
        }
        
        // Initialize channel states
        neuron.channels.ampa_g[c] = 0.0f;
        neuron.channels.ampa_state[c] = 0.0f;
        neuron.channels.nmda_g[c] = 0.0f;
        neuron.channels.nmda_state[c] = 0.0f;
        neuron.channels.gaba_a_g[c] = 0.0f;
        neuron.channels.gaba_a_state[c] = 0.0f;
        neuron.channels.gaba_b_g[c] = 0.0f;
        neuron.channels.gaba_b_state[c] = 0.0f;
        neuron.channels.gaba_b_g_protein[c] = 0.0f;
        
        // Initialize voltage-gated channel states
        neuron.channels.ca_m[c] = 0.05f;  // Initial activation
        neuron.channels.kca_m[c] = 0.05f;
        neuron.channels.hcn_h[c] = 0.05f;
    }
}
```

### 6. Update Constants and Definitions

Extend the constants file with ion channel parameters:

```cpp
// IonChannelConstants.h
#ifndef ION_CHANNEL_CONSTANTS_H
#define ION_CHANNEL_CONSTANTS_H

// Receptor types (same as in Phase 1)
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3

// Calcium dynamics
#define RESTING_CA_CONCENTRATION 0.0001f  // mM
#define MAX_CA_CONCENTRATION 0.01f        // mM
#define CA_BUFFER_CAPACITY 10.0f          // Relative units
#define CA_BUFFER_KD 0.001f               // mM (dissociation constant)
#define CA_EXTRUSION_RATE 0.1f            // Base rate
#define CA_DIFFUSION_RATE 0.1f            // Between compartments

// AMPA receptor parameters
#define AMPA_TAU_RISE 0.5f                // ms
#define AMPA_TAU_DECAY 3.0f               // ms
#define AMPA_REVERSAL 0.0f                // mV

// NMDA receptor parameters
#define NMDA_TAU_RISE 5.0f                // ms
#define NMDA_TAU_DECAY 50.0f              // ms
#define NMDA_REVERSAL 0.0f                // mV
#define NMDA_MG_CONC 1.0f                 // mM
#define NMDA_CA_FRACTION 0.1f             // Fraction of current carried by Ca2+

// GABA-A receptor parameters
#define GABA_A_TAU_RISE 1.0f              // ms
#define GABA_A_TAU_DECAY 7.0f             // ms
#define GABA_A_REVERSAL -70.0f            // mV

// GABA-B receptor parameters
#define GABA_B_TAU_RISE 50.0f             // ms
#define GABA_B_TAU_DECAY 100.0f           // ms
#define GABA_B_TAU_K 10.0f                // ms
#define GABA_B_REVERSAL -90.0f            // mV
