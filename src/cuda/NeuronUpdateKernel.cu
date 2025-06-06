#include "../../include/NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include "NeuronModelConstants.h"
#include <cuda_runtime.h>
#include <math.h>

// Helper functions for Hodgkin-Huxley model
__device__ float alpha_m(float v) {
    float v_shifted = v + 40.0f;
    return (0.1f * v_shifted) / (1.0f - expf(-0.1f * v_shifted));
}

__device__ float beta_m(float v) {
    return 4.0f * expf(-(v + 65.0f) / 18.0f);
}

__device__ float alpha_h(float v) {
    return 0.07f * expf(-(v + 65.0f) / 20.0f);
}

__device__ float beta_h(float v) {
    return 1.0f / (1.0f + expf(-(v + 35.0f) / 10.0f));
}

__device__ float alpha_n(float v) {
    float v_shifted = v + 55.0f;
    return (0.01f * v_shifted) / (1.0f - expf(-0.1f * v_shifted));
}

__device__ float beta_n(float v) {
    return 0.125f * expf(-(v + 65.0f) / 80.0f);
}

// Helper functions for ion channel models
__device__ float computeMgBlock(float v) {
    // Voltage-dependent magnesium block for NMDA receptors
    return 1.0f / (1.0f + 0.28f * expf(-0.062f * v));
}

__device__ float steadyStateActivation(float v, float v_half, float k) {
    // Steady-state activation for voltage-gated channels
    return 1.0f / (1.0f + expf(-(v - v_half) / k));
}

__device__ float calciumDependentActivation(float ca_conc, float ca_half, float hill_coef) {
    // Calcium-dependent activation for KCa channels
    return powf(ca_conc, hill_coef) / (powf(ca_conc, hill_coef) + powf(ca_half, hill_coef));
}

// RK4 integration for Hodgkin-Huxley model with multi-compartment support
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons,
                                     float dt, float current_time, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // 1. Update soma with Hodgkin-Huxley dynamics
    // Extract state variables
    float v = neuron.voltage;
    float m = neuron.m;
    float h = neuron.h;
    float n = neuron.n;
    
    // RK4 integration for soma
    // k1
    float I_Na = HH_G_NA * m*m*m * h * (v - HH_E_NA);
    float I_K = HH_G_K * n*n*n*n * (v - HH_E_K);
    float I_L = HH_G_L * (v - HH_E_L);
    
    // Apply neuromodulatory effects to excitability
    v += neuron.excitability_modulation;
    
    // Apply K+ conductance modulation from neuromodulators
    float k_mod = neuron.k_conductance_modulation;
    
    // Add synaptic currents from ion channels
    float I_syn = 0.0f;
    
    // AMPA current
    float ampa_g = neuron.channels.ampa_g[0];
    float ampa_state = neuron.channels.ampa_state[0];
    I_syn += ampa_g * (v - AMPA_REVERSAL);
    
    // Update AMPA state
    float ampa_dgdt = -ampa_g / AMPA_TAU_DECAY + ampa_state;
    float ampa_dstate_dt = -ampa_state / AMPA_TAU_RISE;
    neuron.channels.ampa_g[0] += ampa_dgdt * dt;
    neuron.channels.ampa_state[0] += ampa_dstate_dt * dt;
    
    // NMDA current with Mg2+ block
    float nmda_g = neuron.channels.nmda_g[0];
    float nmda_state = neuron.channels.nmda_state[0];
    float mg_block = computeMgBlock(v);
    I_syn += nmda_g * mg_block * (v - NMDA_REVERSAL);
    
    // Update NMDA state
    float nmda_dgdt = -nmda_g / NMDA_TAU_DECAY + nmda_state;
    float nmda_dstate_dt = -nmda_state / NMDA_TAU_RISE;
    neuron.channels.nmda_g[0] += nmda_dgdt * dt;
    neuron.channels.nmda_state[0] += nmda_dstate_dt * dt;
    
    // GABA-A current
    float gaba_a_g = neuron.channels.gaba_a_g[0];
    float gaba_a_state = neuron.channels.gaba_a_state[0];
    I_syn += gaba_a_g * (v - GABA_A_REVERSAL);
    
    // Update GABA-A state
    float gaba_a_dgdt = -gaba_a_g / GABA_A_TAU_DECAY + gaba_a_state;
    float gaba_a_dstate_dt = -gaba_a_state / GABA_A_TAU_RISE;
    neuron.channels.gaba_a_g[0] += gaba_a_dgdt * dt;
    neuron.channels.gaba_a_state[0] += gaba_a_dstate_dt * dt;
    
    // GABA-B current
    float gaba_b_g = neuron.channels.gaba_b_g[0];
    float gaba_b_state = neuron.channels.gaba_b_state[0];
    float gaba_b_g_protein = neuron.channels.gaba_b_g_protein[0];
    I_syn += gaba_b_g * (v - GABA_B_REVERSAL);
    
    // Update GABA-B state
    float gaba_b_dg_protein_dt = -gaba_b_g_protein / GABA_B_TAU_DECAY + gaba_b_state;
    float gaba_b_dstate_dt = -gaba_b_state / GABA_B_TAU_RISE;
    float gaba_b_g_inf = powf(gaba_b_g_protein, 4.0f) / (powf(gaba_b_g_protein, 4.0f) + powf(0.5f, 4.0f));
    float gaba_b_dgdt = (gaba_b_g_inf - gaba_b_g) / GABA_B_TAU_K;
    
    neuron.channels.gaba_b_g[0] += gaba_b_dgdt * dt;
    neuron.channels.gaba_b_state[0] += gaba_b_dstate_dt * dt;
    neuron.channels.gaba_b_g_protein[0] += gaba_b_dg_protein_dt * dt;
    
    // Voltage-gated calcium current
    float ca_m = neuron.channels.ca_m[0];
    float ca_m_inf = steadyStateActivation(v, -20.0f, 9.0f);
    float ca_dmdt = (ca_m_inf - ca_m) / 1.0f;
    neuron.channels.ca_m[0] += ca_dmdt * dt;
    float I_Ca = 0.5f * ca_m * ca_m * (v - 50.0f);
    
    // Calcium-dependent potassium current
    float kca_m = neuron.channels.kca_m[0];
    float ca_conc = neuron.ca_conc[0];
    float kca_m_inf = calciumDependentActivation(ca_conc, 0.001f, 4.0f);
    float kca_dmdt = (kca_m_inf - kca_m) / 10.0f;
    neuron.channels.kca_m[0] += kca_dmdt * dt;
    float I_KCa = 0.5f * kca_m * (v - (-90.0f)) * k_mod;
    
    // HCN channel (Ih)
    float hcn_h = neuron.channels.hcn_h[0];
    float hcn_h_inf = 1.0f / (1.0f + expf((v - (-90.0f)) / (-10.0f)));
    float hcn_tau = 50.0f + 450.0f / (1.0f + expf(-(v - (-75.0f)) / 15.0f));
    float hcn_dhdt = (hcn_h_inf - hcn_h) / hcn_tau;
    neuron.channels.hcn_h[0] += hcn_dhdt * dt;
    float I_HCN = 0.1f * hcn_h * (v - (-30.0f));
    
    // Add additional currents to total
    I_syn += I_Ca + I_KCa + I_HCN;
    
    float I_total = -(I_Na + I_K + I_L + I_syn);
    
    float k1_v = dt * I_total;
    float k1_m = dt * (alpha_m(v) * (1.0f - m) - beta_m(v) * m);
    float k1_h = dt * (alpha_h(v) * (1.0f - h) - beta_h(v) * h);
    float k1_n = dt * (alpha_n(v) * (1.0f - n) - beta_n(v) * n);
    
    // k2
    float v2 = v + 0.5f * k1_v;
    float m2 = m + 0.5f * k1_m;
    float h2 = h + 0.5f * k1_h;
    float n2 = n + 0.5f * k1_n;
    
    I_Na = HH_G_NA * m2*m2*m2 * h2 * (v2 - HH_E_NA);
    I_K = HH_G_K * n2*n2*n2*n2 * (v2 - HH_E_K);
    I_L = HH_G_L * (v2 - HH_E_L);
    I_total = -(I_Na + I_K + I_L + I_syn); // Reuse I_syn as approximation
    
    float k2_v = dt * I_total;
    float k2_m = dt * (alpha_m(v2) * (1.0f - m2) - beta_m(v2) * m2);
    float k2_h = dt * (alpha_h(v2) * (1.0f - h2) - beta_h(v2) * h2);
    float k2_n = dt * (alpha_n(v2) * (1.0f - n2) - beta_n(v2) * n2);
    
    // k3
    float v3 = v + 0.5f * k2_v;
    float m3 = m + 0.5f * k2_m;
    float h3 = h + 0.5f * k2_h;
    float n3 = n + 0.5f * k2_n;
    
    I_Na = HH_G_NA * m3*m3*m3 * h3 * (v3 - HH_E_NA);
    I_K = HH_G_K * n3*n3*n3*n3 * (v3 - HH_E_K);
    I_L = HH_G_L * (v3 - HH_E_L);
    I_total = -(I_Na + I_K + I_L + I_syn); // Reuse I_syn as approximation
    
    float k3_v = dt * I_total;
    float k3_m = dt * (alpha_m(v3) * (1.0f - m3) - beta_m(v3) * m3);
    float k3_h = dt * (alpha_h(v3) * (1.0f - h3) - beta_h(v3) * h3);
    float k3_n = dt * (alpha_n(v3) * (1.0f - n3) - beta_n(v3) * n3);
    
    // k4
    float v4 = v + k3_v;
    float m4 = m + k3_m;
    float h4 = h + k3_h;
    float n4 = n + k3_n;
    
    I_Na = HH_G_NA * m4*m4*m4 * h4 * (v4 - HH_E_NA);
    I_K = HH_G_K * n4*n4*n4*n4 * (v4 - HH_E_K);
    I_L = HH_G_L * (v4 - HH_E_L);
    I_total = -(I_Na + I_K + I_L + I_syn); // Reuse I_syn as approximation
    
    float k4_v = dt * I_total;
    float k4_m = dt * (alpha_m(v4) * (1.0f - m4) - beta_m(v4) * m4);
    float k4_h = dt * (alpha_h(v4) * (1.0f - h4) - beta_h(v4) * h4);
    float k4_n = dt * (alpha_n(v4) * (1.0f - n4) - beta_n(v4) * n4);
    
    // Update soma state variables
    v = v + (k1_v + 2.0f*k2_v + 2.0f*k3_v + k4_v) / 6.0f;
    m = m + (k1_m + 2.0f*k2_m + 2.0f*k3_m + k4_m) / 6.0f;
    h = h + (k1_h + 2.0f*k2_h + 2.0f*k3_h + k4_h) / 6.0f;
    n = n + (k1_n + 2.0f*k2_n + 2.0f*k3_n + k4_n) / 6.0f;
    
    // Clamp values to valid ranges
    if (m < 0.0f) m = 0.0f; else if (m > 1.0f) m = 1.0f;
    if (h < 0.0f) h = 0.0f; else if (h > 1.0f) h = 1.0f;
    if (n < 0.0f) n = 0.0f; else if (n > 1.0f) n = 1.0f;
    
    // Store updated soma values
    neuron.voltage = v;
    neuron.m = m;
    neuron.h = h;
    neuron.n = n;
    neuron.voltages[0] = v;
    
    // Check for spike threshold crossing
    if (v > neuron.spike_threshold_modulated && !neuron.spiked) {
        neuron.spiked = true;
        neuron.last_spike_time = current_time;
        
        // Update activity level
        neuron.activity_level = neuron.activity_level * 0.99f + 0.01f;
    }
    
    // 2. Update each dendritic compartment
    for (int c = 1; c < neuron.compartment_count; c++) {
        // Skip inactive compartments
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Extract compartment state variables
        float v_comp = neuron.voltages[c];
        float m_comp = neuron.m_comp[c];
        float h_comp = neuron.h_comp[c];
        float n_comp = neuron.n_comp[c];
        float ca_conc = neuron.ca_conc[c];
        
        // Apply K+ conductance modulation from neuromodulators
        float k_mod_comp = neuron.k_conductance_modulation_dendrites[c];
        
        // Calculate ionic currents based on compartment type
        float I_Na_comp = HH_G_NA * m_comp*m_comp*m_comp * h_comp * (v_comp - HH_E_NA);
        float I_K_comp = HH_G_K * n_comp*n_comp*n_comp*n_comp * (v_comp - HH_E_K) * k_mod_comp;
        float I_L_comp = HH_G_L * (v_comp - HH_E_L);
        
        // Ion channel currents for this compartment
        float I_syn_comp = 0.0f;
        
        // AMPA current
        float ampa_g = neuron.channels.ampa_g[c];
        float ampa_state = neuron.channels.ampa_state[c];
        I_syn_comp += ampa_g * (v_comp - AMPA_REVERSAL);
        
        // Update AMPA state
        float ampa_dgdt = -ampa_g / AMPA_TAU_DECAY + ampa_state;
        float ampa_dstate_dt = -ampa_state / AMPA_TAU_RISE;
        neuron.channels.ampa_g[c] += ampa_dgdt * dt;
        neuron.channels.ampa_state[c] += ampa_dstate_dt * dt;
        
        // NMDA current with Mg2+ block
        float nmda_g = neuron.channels.nmda_g[c];
        float nmda_state = neuron.channels.nmda_state[c];
        float mg_block = computeMgBlock(v_comp);
        I_syn_comp += nmda_g * mg_block * (v_comp - NMDA_REVERSAL);
        
        // Update NMDA state
        float nmda_dgdt = -nmda_g / NMDA_TAU_DECAY + nmda_state;
        float nmda_dstate_dt = -nmda_state / NMDA_TAU_RISE;
        neuron.channels.nmda_g[c] += nmda_dgdt * dt;
        neuron.channels.nmda_state[c] += nmda_dstate_dt * dt;
        
        // NMDA-mediated calcium influx
        if (nmda_g > 0.0f) {
            float ca_influx = nmda_g * mg_block * NMDA_CA_FRACTION * neuron.ca_influx_modulation[c];
            neuron.ca_conc[c] += ca_influx * dt;
        }
        
        // GABA-A current
        float gaba_a_g = neuron.channels.gaba_a_g[c];
        float gaba_a_state = neuron.channels.gaba_a_state[c];
        I_syn_comp += gaba_a_g * (v_comp - GABA_A_REVERSAL);
        
        // Update GABA-A state
        float gaba_a_dgdt = -gaba_a_g / GABA_A_TAU_DECAY + gaba_a_state;
        float gaba_a_dstate_dt = -gaba_a_state / GABA_A_TAU_RISE;
        neuron.channels.gaba_a_g[c] += gaba_a_dgdt * dt;
        neuron.channels.gaba_a_state[c] += gaba_a_dstate_dt * dt;
        
        // GABA-B current
        float gaba_b_g = neuron.channels.gaba_b_g[c];
        float gaba_b_state = neuron.channels.gaba_b_state[c];
        float gaba_b_g_protein = neuron.channels.gaba_b_g_protein[c];
        I_syn_comp += gaba_b_g * (v_comp - GABA_B_REVERSAL);
        
        // Update GABA-B state
        float gaba_b_dg_protein_dt = -gaba_b_g_protein / GABA_B_TAU_DECAY + gaba_b_state;
        float gaba_b_dstate_dt = -gaba_b_state / GABA_B_TAU_RISE;
        float gaba_b_g_inf = powf(gaba_b_g_protein, 4.0f) / (powf(gaba_b_g_protein, 4.0f) + powf(0.5f, 4.0f));
        float gaba_b_dgdt = (gaba_b_g_inf - gaba_b_g) / GABA_B_TAU_K;
        
        neuron.channels.gaba_b_g[c] += gaba_b_dgdt * dt;
        neuron.channels.gaba_b_state[c] += gaba_b_dstate_dt * dt;
        neuron.channels.gaba_b_g_protein[c] += gaba_b_dg_protein_dt * dt;
        
        // Additional voltage-gated channels based on compartment type
        float I_additional = 0.0f;
        
        if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
            // Voltage-gated calcium current (stronger in apical dendrites)
            float ca_m = neuron.channels.ca_m[c];
            float ca_m_inf = steadyStateActivation(v_comp, -20.0f, 9.0f);
            float ca_dmdt = (ca_m_inf - ca_m) / 1.0f;
            neuron.channels.ca_m[c] += ca_dmdt * dt;
            float I_Ca = 0.8f * ca_m * ca_m * (v_comp - 50.0f);
            I_additional += I_Ca;
            
            // Calcium influx from voltage-gated calcium channels
            if (ca_m > 0.1f) {
                float ca_influx = -I_Ca * 0.01f * neuron.ca_influx_modulation[c];
                neuron.ca_conc[c] += ca_influx * dt;
            }
        } else {
            // Voltage-gated calcium current (weaker in basal dendrites)
            float ca_m = neuron.channels.ca_m[c];
            float ca_m_inf = steadyStateActivation(v_comp, -20.0f, 9.0f);
            float ca_dmdt = (ca_m_inf - ca_m) / 1.0f;
            neuron.channels.ca_m[c] += ca_dmdt * dt;
            float I_Ca = 0.3f * ca_m * ca_m * (v_comp - 50.0f);
            I_additional += I_Ca;
            
            // Calcium influx from voltage-gated calcium channels
            if (ca_m > 0.1f) {
                float ca_influx = -I_Ca * 0.005f * neuron.ca_influx_modulation[c];
                neuron.ca_conc[c] += ca_influx * dt;
            }
        }
        
        // Calcium-dependent potassium current
        float kca_m = neuron.channels.kca_m[c];
        float kca_m_inf = calciumDependentActivation(ca_conc, 0.001f, 4.0f);
        float kca_dmdt = (kca_m_inf - kca_m) / 10.0f;
        neuron.channels.kca_m[c] += kca_dmdt * dt;
        float I_KCa = 0.5f * kca_m * (v_comp - (-90.0f)) * k_mod_comp;
        I_additional += I_KCa;
        
        // HCN channel (Ih)
        float hcn_h = neuron.channels.hcn_h[c];
        float hcn_h_inf = 1.0f / (1.0f + expf((v_comp - (-90.0f)) / (-10.0f)));
        float hcn_tau = 50.0f + 450.0f / (1.0f + expf(-(v_comp - (-75.0f)) / 15.0f));
        float hcn_dhdt = (hcn_h_inf - hcn_h) / hcn_tau;
        neuron.channels.hcn_h[c] += hcn_dhdt * dt;
        float I_HCN = 0.1f * hcn_h * (v_comp - (-30.0f));
        I_additional += I_HCN;
        
        // Add additional currents to synaptic current
        I_syn_comp += I_additional;
        
        // Calculate coupling current to parent compartment
        int parent = neuron.parent_compartment[c];
        float I_coupling = 0.0f;
        if (parent >= 0) {
            I_coupling = neuron.coupling_conductance[c] * (neuron.voltages[parent] - v_comp);
        }
        
        // Total current for this compartment
        float I_total_comp = -(I_Na_comp + I_K_comp + I_L_comp + I_syn_comp) + I_coupling;
        
        // Simplified RK4 for compartment (just for voltage)
        float k1 = dt * I_total_comp;
        float k2 = dt * I_total_comp; // Approximation
        float k3 = dt * I_total_comp; // Approximation
        float k4 = dt * I_total_comp; // Approximation
        
        // Update compartment voltage
        v_comp = v_comp + (k1 + 2.0f*k2 + 2.0f*k3 + k4) / 6.0f;
        
        // Update gating variables (simplified)
        m_comp = m_comp + dt * (alpha_m(v_comp) * (1.0f - m_comp) - beta_m(v_comp) * m_comp);
        h_comp = h_comp + dt * (alpha_h(v_comp) * (1.0f - h_comp) - beta_h(v_comp) * h_comp);
        n_comp = n_comp + dt * (alpha_n(v_comp) * (1.0f - n_comp) - beta_n(v_comp) * n_comp);
        
        // Clamp values
        if (m_comp < 0.0f) m_comp = 0.0f; else if (m_comp > 1.0f) m_comp = 1.0f;
        if (h_comp < 0.0f) h_comp = 0.0f; else if (h_comp > 1.0f) h_comp = 1.0f;
        if (n_comp < 0.0f) n_comp = 0.0f; else if (n_comp > 1.0f) n_comp = 1.0f;
        
        // Check for dendritic spike
        if (v_comp > DENDRITIC_SPIKE_THRESHOLD && !neuron.dendritic_spike[c]) {
            neuron.dendritic_spike[c] = true;
            neuron.last_dendritic_spike[c] = current_time;
            
            // Propagate effect to soma and parent compartments
            if (parent >= 0) {
                neuron.voltages[parent] += DENDRITIC_SPIKE_PROPAGATION_STRENGTH;
            }
            
            // For apical dendrites, trigger calcium influx
            if (neuron.compartment_types[c] == COMPARTMENT_APICAL) {
                neuron.ca_conc[c] += DENDRITIC_SPIKE_CA_INFLUX;
            }
        }
        
        // Update calcium dynamics
        // Calcium buffering
        float free_buffer = CA_BUFFER_CAPACITY - neuron.ca_buffer[c];
        float buffer_forward_rate = 0.5f;
        float buffer_reverse_rate = 0.1f;
        float buffering = buffer_forward_rate * ca_conc * free_buffer - buffer_reverse_rate * neuron.ca_buffer[c];
        
        // Calcium extrusion (pump)
        float pump_rate = neuron.ca_pump_rate[c];
        float extrusion = pump_rate * ca_conc / (ca_conc + 0.0001f);
        
        // Calcium diffusion between compartments
        float diffusion = 0.0f;
        if (parent >= 0) {
            float ca_parent = neuron.ca_conc[parent];
            diffusion = CA_DIFFUSION_RATE * (ca_parent - ca_conc);
        }
        
        // Update calcium concentration
        float dca_dt = -buffering - extrusion + diffusion;
        float dca_buffer_dt = buffering;
        
        neuron.ca_conc[c] += dca_dt * dt;
        neuron.ca_buffer[c] += dca_buffer_dt * dt;
        
        // Ensure calcium stays in valid range
        if (neuron.ca_conc[c] < 0.0f) neuron.ca_conc[c] = 0.0f;
        if (neuron.ca_buffer[c] < 0.0f) neuron.ca_buffer[c] = 0.0f;
        if (neuron.ca_buffer[c] > CA_BUFFER_CAPACITY) neuron.ca_buffer[c] = CA_BUFFER_CAPACITY;
        
        // Store updated compartment state
        neuron.voltages[c] = v_comp;
        neuron.m_comp[c] = m_comp;
        neuron.h_comp[c] = h_comp;
        neuron.n_comp[c] = n_comp;
    }
}

__global__ void updateNeuronVoltages(GPUNeuronState* neurons,
                                    float* I_leak, float* Cm,
                                    float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Skip inactive neurons
    if (neurons[idx].active == 0) return;
    
    // Simple voltage update for each compartment
    for (int c = 0; c < neurons[idx].compartment_count; c++) {
        float I_leak_val = (I_leak != nullptr) ? I_leak[idx * MAX_COMPARTMENTS + c] : neurons[idx].I_leak[c];
        float Cm_val = (Cm != nullptr) ? Cm[idx * MAX_COMPARTMENTS + c] : neurons[idx].Cm[c];
        
        if (Cm_val > 0.0f) {
            neurons[idx].voltages[c] += dt * I_leak_val / Cm_val;
        }
    }
    
    // Update main voltage
    neurons[idx].voltage = neurons[idx].voltages[0];
}

// Update activity levels for neurons
__global__ void updateActivityLevels(GPUNeuronState* neurons, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Decay activity level
    neuron.activity_level *= expf(-dt / ACTIVITY_TAU);
}

// Process dendritic spikes and their effects
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
                    neuron.ca_conc[c] += DENDRITIC_SPIKE_CA_INFLUX * 0.1f; // Smaller continuous influx
                }
                
                // Affect parent compartments
                int parent = neuron.parent_compartment[c];
                if (parent >= 0) {
                    // Depolarize parent compartment
                    neuron.voltages[parent] += DENDRITIC_SPIKE_PROPAGATION_STRENGTH * 0.1f;
                }
            }
        }
    }
}
