#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Enhanced RK4 neuron update kernel with comprehensive ion channel dynamics
 * Integrates AMPA, NMDA, GABA-A, GABA-B, voltage-gated Ca, KCa, and HCN channels
 */
__global__ void enhancedRK4NeuronUpdateKernel(
    GPUNeuronState* neurons,
    float dt,
    float current_time,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Initialize channel models for this neuron
    AMPAChannel ampa_channel;
    NMDAChannel nmda_channel;  
    GABAA_Channel gaba_a_channel;
    GABAB_Channel gaba_b_channel;
    CaChannel ca_channel;
    KCaChannel kca_channel;
    HCNChannel hcn_channel;
    CalciumDynamics ca_dynamics;
    
    // Process each compartment
    for (int c = 0; c < neuron.compartment_count; c++) {
        // Skip inactive compartments
        if (neuron.compartment_types[c] == COMPARTMENT_INACTIVE) continue;
        
        // Initialize channel parameters based on compartment type
        initializeChannelParameters(
            neuron.compartment_types[c],
            &ampa_channel, &nmda_channel, &gaba_a_channel, &gaba_b_channel,
            &ca_channel, &kca_channel, &hcn_channel, &ca_dynamics
        );
        
        // Get current states
        float v = (c == 0) ? neuron.voltage : neuron.voltages[c];
        float m = (c == 0) ? neuron.m : neuron.m_comp[c];
        float h = (c == 0) ? neuron.h : neuron.h_comp[c];
        float n = (c == 0) ? neuron.n : neuron.n_comp[c];
        
        // Ion channel states
        float ampa_g = neuron.channels.ampa_g[c];
        float ampa_state = neuron.channels.ampa_state[c];
        float nmda_g = neuron.channels.nmda_g[c];
        float nmda_state = neuron.channels.nmda_state[c];
        float gaba_a_g = neuron.channels.gaba_a_g[c];
        float gaba_a_state = neuron.channels.gaba_a_state[c];
        float gaba_b_g = neuron.channels.gaba_b_g[c];
        float gaba_b_state = neuron.channels.gaba_b_state[c];
        float gaba_b_g_protein = neuron.channels.gaba_b_g_protein[c];
        
        // Voltage-gated channel states
        float ca_m = neuron.channels.ca_m[c];
        float kca_m = neuron.channels.kca_m[c];
        float hcn_h = neuron.channels.hcn_h[c];
        
        // Calcium states
        float ca_conc = neuron.ca_conc[c];
        float ca_buffer = neuron.ca_buffer[c];
        
        // ========================================
        // RK4 INTEGRATION - K1
        // ========================================
        float k1_v, k1_m, k1_h, k1_n;
        float k1_ampa_g, k1_ampa_state, k1_nmda_g, k1_nmda_state;
        float k1_gaba_a_g, k1_gaba_a_state, k1_gaba_b_g, k1_gaba_b_state, k1_gaba_b_g_protein;
        float k1_ca_m, k1_kca_m, k1_hcn_h, k1_ca_conc, k1_ca_buffer;
        
        computeDerivatives(
            v, m, h, n, ca_conc, ca_buffer,
            ampa_g, ampa_state, nmda_g, nmda_state,
            gaba_a_g, gaba_a_state, gaba_b_g, gaba_b_state, gaba_b_g_protein,
            ca_m, kca_m, hcn_h,
            &ampa_channel, &nmda_channel, &gaba_a_channel, &gaba_b_channel,
            &ca_channel, &kca_channel, &hcn_channel, &ca_dynamics,
            current_time, c, neuron,
            &k1_v, &k1_m, &k1_h, &k1_n,
            &k1_ampa_g, &k1_ampa_state, &k1_nmda_g, &k1_nmda_state,
            &k1_gaba_a_g, &k1_gaba_a_state, &k1_gaba_b_g, &k1_gaba_b_state, &k1_gaba_b_g_protein,
            &k1_ca_m, &k1_kca_m, &k1_hcn_h, &k1_ca_conc, &k1_ca_buffer
        );
        
        // ========================================
        // RK4 INTEGRATION - K2
        // ========================================
        float k2_v, k2_m, k2_h, k2_n;
        float k2_ampa_g, k2_ampa_state, k2_nmda_g, k2_nmda_state;
        float k2_gaba_a_g, k2_gaba_a_state, k2_gaba_b_g, k2_gaba_b_state, k2_gaba_b_g_protein;
        float k2_ca_m, k2_kca_m, k2_hcn_h, k2_ca_conc, k2_ca_buffer;
        
        computeDerivatives(
            v + 0.5f * dt * k1_v, m + 0.5f * dt * k1_m, h + 0.5f * dt * k1_h, n + 0.5f * dt * k1_n,
            ca_conc + 0.5f * dt * k1_ca_conc, ca_buffer + 0.5f * dt * k1_ca_buffer,
            ampa_g + 0.5f * dt * k1_ampa_g, ampa_state + 0.5f * dt * k1_ampa_state,
            nmda_g + 0.5f * dt * k1_nmda_g, nmda_state + 0.5f * dt * k1_nmda_state,
            gaba_a_g + 0.5f * dt * k1_gaba_a_g, gaba_a_state + 0.5f * dt * k1_gaba_a_state,
            gaba_b_g + 0.5f * dt * k1_gaba_b_g, gaba_b_state + 0.5f * dt * k1_gaba_b_state,
            gaba_b_g_protein + 0.5f * dt * k1_gaba_b_g_protein,
            ca_m + 0.5f * dt * k1_ca_m, kca_m + 0.5f * dt * k1_kca_m, hcn_h + 0.5f * dt * k1_hcn_h,
            &ampa_channel, &nmda_channel, &gaba_a_channel, &gaba_b_channel,
            &ca_channel, &kca_channel, &hcn_channel, &ca_dynamics,
            current_time + 0.5f * dt, c, neuron,
            &k2_v, &k2_m, &k2_h, &k2_n,
            &k2_ampa_g, &k2_ampa_state, &k2_nmda_g, &k2_nmda_state,
            &k2_gaba_a_g, &k2_gaba_a_state, &k2_gaba_b_g, &k2_gaba_b_state, &k2_gaba_b_g_protein,
            &k2_ca_m, &k2_kca_m, &k2_hcn_h, &k2_ca_conc, &k2_ca_buffer
        );
        
        // ========================================
        // RK4 INTEGRATION - K3
        // ========================================
        float k3_v, k3_m, k3_h, k3_n;
        float k3_ampa_g, k3_ampa_state, k3_nmda_g, k3_nmda_state;
        float k3_gaba_a_g, k3_gaba_a_state, k3_gaba_b_g, k3_gaba_b_state, k3_gaba_b_g_protein;
        float k3_ca_m, k3_kca_m, k3_hcn_h, k3_ca_conc, k3_ca_buffer;
        
        computeDerivatives(
            v + 0.5f * dt * k2_v, m + 0.5f * dt * k2_m, h + 0.5f * dt * k2_h, n + 0.5f * dt * k2_n,
            ca_conc + 0.5f * dt * k2_ca_conc, ca_buffer + 0.5f * dt * k2_ca_buffer,
            ampa_g + 0.5f * dt * k2_ampa_g, ampa_state + 0.5f * dt * k2_ampa_state,
            nmda_g + 0.5f * dt * k2_nmda_g, nmda_state + 0.5f * dt * k2_nmda_state,
            gaba_a_g + 0.5f * dt * k2_gaba_a_g, gaba_a_state + 0.5f * dt * k2_gaba_a_state,
            gaba_b_g + 0.5f * dt * k2_gaba_b_g, gaba_b_state + 0.5f * dt * k2_gaba_b_state,
            gaba_b_g_protein + 0.5f * dt * k2_gaba_b_g_protein,
            ca_m + 0.5f * dt * k2_ca_m, kca_m + 0.5f * dt * k2_kca_m, hcn_h + 0.5f * dt * k2_hcn_h,
            &ampa_channel, &nmda_channel, &gaba_a_channel, &gaba_b_channel,
            &ca_channel, &kca_channel, &hcn_channel, &ca_dynamics,
            current_time + 0.5f * dt, c, neuron,
            &k3_v, &k3_m, &k3_h, &k3_n,
            &k3_ampa_g, &k3_ampa_state, &k3_nmda_g, &k3_nmda_state,
            &k3_gaba_a_g, &k3_gaba_a_state, &k3_gaba_b_g, &k3_gaba_b_state, &k3_gaba_b_g_protein,
            &k3_ca_m, &k3_kca_m, &k3_hcn_h, &k3_ca_conc, &k3_ca_buffer
        );
        
        // ========================================
        // RK4 INTEGRATION - K4
        // ========================================
        float k4_v, k4_m, k4_h, k4_n;
        float k4_ampa_g, k4_ampa_state, k4_nmda_g, k4_nmda_state;
        float k4_gaba_a_g, k4_gaba_a_state, k4_gaba_b_g, k4_gaba_b_state, k4_gaba_b_g_protein;
        float k4_ca_m, k4_kca_m, k4_hcn_h, k4_ca_conc, k4_ca_buffer;
        
        computeDerivatives(
            v + dt * k3_v, m + dt * k3_m, h + dt * k3_h, n + dt * k3_n,
            ca_conc + dt * k3_ca_conc, ca_buffer + dt * k3_ca_buffer,
            ampa_g + dt * k3_ampa_g, ampa_state + dt * k3_ampa_state,
            nmda_g + dt * k3_nmda_g, nmda_state + dt * k3_nmda_state,
            gaba_a_g + dt * k3_gaba_a_g, gaba_a_state + dt * k3_gaba_a_state,
            gaba_b_g + dt * k3_gaba_b_g, gaba_b_state + dt * k3_gaba_b_state,
            gaba_b_g_protein + dt * k3_gaba_b_g_protein,
            ca_m + dt * k3_ca_m, kca_m + dt * k3_kca_m, hcn_h + dt * k3_hcn_h,
            &ampa_channel, &nmda_channel, &gaba_a_channel, &gaba_b_channel,
            &ca_channel, &kca_channel, &hcn_channel, &ca_dynamics,
            current_time + dt, c, neuron,
            &k4_v, &k4_m, &k4_h, &k4_n,
            &k4_ampa_g, &k4_ampa_state, &k4_nmda_g, &k4_nmda_state,
            &k4_gaba_a_g, &k4_gaba_a_state, &k4_gaba_b_g, &k4_gaba_b_state, &k4_gaba_b_g_protein,
            &k4_ca_m, &k4_kca_m, &k4_hcn_h, &k4_ca_conc, &k4_ca_buffer
        );
        
        // ========================================
        // FINAL RK4 UPDATE
        // ========================================
        v += dt * (k1_v + 2.0f*k2_v + 2.0f*k3_v + k4_v) / 6.0f;
        m += dt * (k1_m + 2.0f*k2_m + 2.0f*k3_m + k4_m) / 6.0f;
        h += dt * (k1_h + 2.0f*k2_h + 2.0f*k3_h + k4_h) / 6.0f;
        n += dt * (k1_n + 2.0f*k2_n + 2.0f*k3_n + k4_n) / 6.0f;
        
        // Update ion channel states
        ampa_g += dt * (k1_ampa_g + 2.0f*k2_ampa_g + 2.0f*k3_ampa_g + k4_ampa_g) / 6.0f;
        ampa_state += dt * (k1_ampa_state + 2.0f*k2_ampa_state + 2.0f*k3_ampa_state + k4_ampa_state) / 6.0f;
        nmda_g += dt * (k1_nmda_g + 2.0f*k2_nmda_g + 2.0f*k3_nmda_g + k4_nmda_g) / 6.0f;
        nmda_state += dt * (k1_nmda_state + 2.0f*k2_nmda_state + 2.0f*k3_nmda_state + k4_nmda_state) / 6.0f;
        gaba_a_g += dt * (k1_gaba_a_g + 2.0f*k2_gaba_a_g + 2.0f*k3_gaba_a_g + k4_gaba_a_g) / 6.0f;
        gaba_a_state += dt * (k1_gaba_a_state + 2.0f*k2_gaba_a_state + 2.0f*k3_gaba_a_state + k4_gaba_a_state) / 6.0f;
        gaba_b_g += dt * (k1_gaba_b_g + 2.0f*k2_gaba_b_g + 2.0f*k3_gaba_b_g + k4_gaba_b_g) / 6.0f;
        gaba_b_state += dt * (k1_gaba_b_state + 2.0f*k2_gaba_b_state + 2.0f*k3_gaba_b_state + k4_gaba_b_state) / 6.0f;
        gaba_b_g_protein += dt * (k1_gaba_b_g_protein + 2.0f*k2_gaba_b_g_protein + 2.0f*k3_gaba_b_g_protein + k4_gaba_b_g_protein) / 6.0f;
        
        // Update voltage-gated channel states
        ca_m += dt * (k1_ca_m + 2.0f*k2_ca_m + 2.0f*k3_ca_m + k4_ca_m) / 6.0f;
        kca_m += dt * (k1_kca_m + 2.0f*k2_kca_m + 2.0f*k3_kca_m + k4_kca_m) / 6.0f;
        hcn_h += dt * (k1_hcn_h + 2.0f*k2_hcn_h + 2.0f*k3_hcn_h + k4_hcn_h) / 6.0f;
        
        // Update calcium dynamics
        ca_conc += dt * (k1_ca_conc + 2.0f*k2_ca_conc + 2.0f*k3_ca_conc + k4_ca_conc) / 6.0f;
        ca_buffer += dt * (k1_ca_buffer + 2.0f*k2_ca_buffer + 2.0f*k3_ca_buffer + k4_ca_buffer) / 6.0f;
        
        // ========================================
        // ENFORCE BOUNDS AND STABILITY
        // ========================================
        // Clamp Hodgkin-Huxley variables
        m = fmaxf(0.0f, fminf(1.0f, m));
        h = fmaxf(0.0f, fminf(1.0f, h));
        n = fmaxf(0.0f, fminf(1.0f, n));
        
        // Clamp conductances
        ampa_g = fmaxf(0.0f, fminf(MAX_CONDUCTANCE, ampa_g));
        ampa_state = fmaxf(0.0f, ampa_state);
        nmda_g = fmaxf(0.0f, fminf(MAX_CONDUCTANCE, nmda_g));
        nmda_state = fmaxf(0.0f, nmda_state);
        gaba_a_g = fmaxf(0.0f, fminf(MAX_CONDUCTANCE, gaba_a_g));
        gaba_a_state = fmaxf(0.0f, gaba_a_state);
        gaba_b_g = fmaxf(0.0f, fminf(MAX_CONDUCTANCE, gaba_b_g));
        gaba_b_state = fmaxf(0.0f, gaba_b_state);
        gaba_b_g_protein = fmaxf(0.0f, fminf(1.0f, gaba_b_g_protein));
        
        // Clamp voltage-gated channel variables
        ca_m = fmaxf(0.0f, fminf(1.0f, ca_m));
        kca_m = fmaxf(0.0f, fminf(1.0f, kca_m));
        hcn_h = fmaxf(0.0f, fminf(1.0f, hcn_h));
        
        // Clamp calcium concentrations
        ca_conc = fmaxf(MIN_CA_CONCENTRATION, fminf(MAX_CA_CONCENTRATION, ca_conc));
        ca_buffer = fmaxf(0.0f, ca_buffer);
        
        // ========================================
        // STORE UPDATED VALUES
        // ========================================
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
        
        // Store ion channel states
        neuron.channels.ampa_g[c] = ampa_g;
        neuron.channels.ampa_state[c] = ampa_state;
        neuron.channels.nmda_g[c] = nmda_g;
        neuron.channels.nmda_state[c] = nmda_state;
        neuron.channels.gaba_a_g[c] = gaba_a_g;
        neuron.channels.gaba_a_state[c] = gaba_a_state;
        neuron.channels.gaba_b_g[c] = gaba_b_g;
        neuron.channels.gaba_b_state[c] = gaba_b_state;
        neuron.channels.gaba_b_g_protein[c] = gaba_b_g_protein;
        
        // Store voltage-gated channel states
        neuron.channels.ca_m[c] = ca_m;
        neuron.channels.kca_m[c] = kca_m;
        neuron.channels.hcn_h[c] = hcn_h;
        
        // Store calcium states
        neuron.ca_conc[c] = ca_conc;
        neuron.ca_buffer[c] = ca_buffer;
        
        // ========================================
        // SPIKE DETECTION
        // ========================================
        if (c == 0) {  // Check soma for action potential
            if (v > neuron.spike_threshold_modulated && !neuron.spiked) {
                neuron.spiked = true;
                neuron.last_spike_time = current_time;
                neuron.spike_count++;
                
                // Update activity level with decay
                neuron.activity_level = neuron.activity_level * 0.99f + 0.01f;
                
                // Calculate firing rate (simple moving average)
                float time_since_last = current_time - neuron.last_spike_time;
                if (time_since_last > 0.0f) {
                    neuron.avg_firing_rate = 0.9f * neuron.avg_firing_rate + 0.1f * (1000.0f / time_since_last);
                }
            }
        }
        
        // Check for dendritic spikes
        if (c > 0 && v > neuron.dendritic_threshold[c]) {
            if (!neuron.dendritic_spike[c]) {
                neuron.dendritic_spike[c] = true;
                neuron.dendritic_spike_time[c] = current_time;
            }
        } else {
            neuron.dendritic_spike[c] = false;
        }
    }
    
    // Update neuron-level metrics
    neuron.last_update_time = current_time;
    neuron.time_since_spike = current_time - neuron.last_spike_time;
    
    // Calculate total synaptic currents for monitoring
    float total_excitatory = 0.0f;
    float total_inhibitory = 0.0f;
    
    for (int c = 0; c < neuron.compartment_count; c++) {
        if (neuron.compartment_types[c] != COMPARTMENT_INACTIVE) {
            float v_comp = (c == 0) ? neuron.voltage : neuron.voltages[c];
            total_excitatory += neuron.channels.ampa_g[c] * (v_comp - AMPA_REVERSAL);
            total_excitatory += neuron.channels.nmda_g[c] * computeMgBlock(v_comp) * (v_comp - NMDA_REVERSAL);
            total_inhibitory += neuron.channels.gaba_a_g[c] * (v_comp - GABA_A_REVERSAL);
            total_inhibitory += neuron.channels.gaba_b_g[c] * neuron.channels.gaba_b_g_protein[c] * (v_comp - GABA_B_REVERSAL);
        }
    }
    
    neuron.total_excitatory_input = total_excitatory;
    neuron.total_inhibitory_input = total_inhibitory;
    neuron.total_current_injected = total_excitatory + total_inhibitory;
    
    // Update energy consumption (simplified model)
    neuron.energy_consumption += (fabs(total_excitatory) + fabs(total_inhibitory)) * dt * 0.001f;
}

/**
 * Device function to compute derivatives for all state variables
 */
__device__ void computeDerivatives(
    float v, float m, float h, float n, float ca_conc, float ca_buffer,
    float ampa_g, float ampa_state, float nmda_g, float nmda_state,
    float gaba_a_g, float gaba_a_state, float gaba_b_g, float gaba_b_state, float gaba_b_g_protein,
    float ca_m, float kca_m, float hcn_h,
    AMPAChannel* ampa_channel, NMDAChannel* nmda_channel, 
    GABAA_Channel* gaba_a_channel, GABAB_Channel* gaba_b_channel,
    CaChannel* ca_channel, KCaChannel* kca_channel, HCNChannel* hcn_channel,
    CalciumDynamics* ca_dynamics,
    float current_time, int compartment_idx, const GPUNeuronState& neuron,
    float* dv_dt, float* dm_dt, float* dh_dt, float* dn_dt,
    float* dampa_g_dt, float* dampa_state_dt, float* dnmda_g_dt, float* dnmda_state_dt,
    float* dgaba_a_g_dt, float* dgaba_a_state_dt, float* dgaba_b_g_dt, float* dgaba_b_state_dt, float* dgaba_b_g_protein_dt,
    float* dca_m_dt, float* dkca_m_dt, float* dhcn_h_dt, float* dca_conc_dt, float* dca_buffer_dt
) {
    // ========================================
    // COMPUTE IONIC CURRENTS
    // ========================================
    
    // Hodgkin-Huxley currents
    float I_Na = HH_G_NA * m*m*m * h * (v - HH_E_NA);
    float I_K = HH_G_K * n*n*n*n * (v - HH_E_K);
    float I_L = HH_G_L * (v - HH_E_L);
    
    // Synaptic currents
    float I_AMPA = ampa_channel->computeCurrent(v, ampa_g);
    float I_NMDA = nmda_channel->computeCurrent(v, nmda_g);
    float I_GABA_A = gaba_a_channel->computeCurrent(v, gaba_a_g);
    float I_GABA_B = gaba_b_channel->computeCurrent(v, gaba_b_g, gaba_b_g_protein);
    
    // Voltage-gated currents
    float I_Ca = ca_channel->computeCurrent(v, ca_m);
    float I_KCa = kca_channel->computeCurrent(v, kca_m);
    float I_HCN = hcn_channel->computeCurrent(v, hcn_h);
    
    // Coupling current from parent compartment
    float I_coupling = 0.0f;
    if (compartment_idx > 0) {
        int parent = neuron.parent_compartment[compartment_idx];
        if (parent >= 0) {
            float v_parent = (parent == 0) ? neuron.voltage : neuron.voltages[parent];
            I_coupling = neuron.coupling_conductance[compartment_idx] * (v_parent - v);
        }
    }
    
    // Total current
    float I_total = -(I_Na + I_K + I_L + I_AMPA + I_NMDA + I_GABA_A + I_GABA_B + I_Ca + I_KCa + I_HCN) + I_coupling;
    
    // ========================================
    // COMPUTE CALCIUM CURRENT FOR DYNAMICS
    // ========================================
    float I_Ca_total = -I_Ca;  // Calcium influx from voltage-gated channels
    
    // Add calcium component from NMDA
    if (nmda_g > 0.0f) {
        I_Ca_total += -nmda_channel->computeCalciumCurrent(v, nmda_g);
    }
    
    // ========================================
    // COMPUTE STATE VARIABLE DERIVATIVES
    // ========================================
    
    // Voltage derivative
    *dv_dt = I_total / neuron.membrane_capacitance;
    
    // Hodgkin-Huxley derivatives
    *dm_dt = alpha_m(v) * (1.0f - m) - beta_m(v) * m;
    *dh_dt = alpha_h(v) * (1.0f - h) - beta_h(v) * h;
    *dn_dt = alpha_n(v) * (1.0f - n) - beta_n(v) * n;
    
    // Synaptic channel derivatives (with synaptic input)
    float synaptic_input = 0.0f;  // This would be set by synaptic input processing
    
    *dampa_g_dt = -ampa_g / ampa_channel->tau_decay + ampa_state;
    *dampa_state_dt = -ampa_state / ampa_channel->tau_rise + synaptic_input;
    
    *dnmda_g_dt = -nmda_g / nmda_channel->tau_decay + nmda_state;
    *dnmda_state_dt = -nmda_state / nmda_channel->tau_rise + synaptic_input;
    
    *dgaba_a_g_dt = -gaba_a_g / gaba_a_channel->tau_decay + gaba_a_state;
    *dgaba_a_state_dt = -gaba_a_state / gaba_a_channel->tau_rise + synaptic_input;
    
    *dgaba_b_g_dt = -gaba_b_g / gaba_b_channel->tau_decay + gaba_b_state;
    *dgaba_b_state_dt = -gaba_b_state / gaba_b_channel->tau_rise + synaptic_input;
    *dgaba_b_g_protein_dt = (gaba_b_g - gaba_b_g_protein) / gaba_b_channel->tau_k;
    
    // Voltage-gated channel derivatives
    float ca_m_inf = ca_channel->steadyStateActivation(v);
    *dca_m_dt = (ca_m_inf - ca_m) / ca_channel->tau_act;
    
    float kca_m_inf = kca_channel->calciumDependentActivation(ca_conc);
    *dkca_m_dt = (kca_m_inf - kca_m) / kca_channel->tau_act;
    
    float hcn_h_inf = hcn_channel->steadyStateActivation(v);
    float hcn_tau = hcn_channel->activationTimeConstant(v);
    *dhcn_h_dt = (hcn_h_inf - hcn_h) / hcn_tau;
    
    // Calcium dynamics derivatives
    float ca_influx = -I_Ca_total * ca_dynamics->volume_factor;
    float ca_extrusion = ca_dynamics->extrusion_rate * (ca_conc - ca_dynamics->resting_ca);
    float buffer_binding = ca_dynamics->computeBuffering(ca_conc, ca_buffer);
    
    *dca_conc_dt = ca_influx - ca_extrusion - buffer_binding;
    *dca_buffer_dt = buffer_binding;
}

/**
 * Device function to initialize channel parameters based on compartment type
 */
__device__ void initializeChannelParameters(
    int compartment_type,
    AMPAChannel* ampa, NMDAChannel* nmda, 
    GABAA_Channel* gaba_a, GABAB_Channel* gaba_b,
    CaChannel* ca, KCaChannel* kca, HCNChannel* hcn,
    CalciumDynamics* ca_dyn
) {
    // Initialize AMPA channel
    ampa->g_max = (compartment_type == COMPARTMENT_SOMA) ? AMPA_G_MAX_SOMA : AMPA_G_MAX_DENDRITE;
    ampa->tau_rise = AMPA_TAU_RISE;
    ampa->tau_decay = AMPA_TAU_DECAY;
    ampa->reversal = AMPA_REVERSAL;
    
    // Initialize NMDA channel
    nmda->g_max = (compartment_type == COMPARTMENT_SOMA) ? NMDA_G_MAX_SOMA : NMDA_G_MAX_DENDRITE;
    nmda->tau_rise = NMDA_TAU_RISE;
    nmda->tau_decay = NMDA_TAU_DECAY;
    nmda->reversal = NMDA_REVERSAL;
    nmda->mg_conc = NMDA_MG_CONC;
    nmda->ca_fraction = NMDA_CA_FRACTION;
    
    // Initialize GABA-A channel
    gaba_a->g_max = (compartment_type == COMPARTMENT_SOMA) ? GABA_A_G_MAX_SOMA : GABA_A_G_MAX_DENDRITE;
    gaba_a->tau_rise = GABA_A_TAU_RISE;
    gaba_a->tau_decay = GABA_A_TAU_DECAY;
    gaba_a->reversal = GABA_A_REVERSAL;
    
    // Initialize GABA-B channel
    gaba_b->g_max = (compartment_type == COMPARTMENT_SOMA) ? GABA_B_G_MAX_SOMA : GABA_B_G_MAX_DENDRITE;
    gaba_b->tau_rise = GABA_B_TAU_RISE;
    gaba_b->tau_decay = GABA_B_TAU_DECAY;
    gaba_b->tau_k = GABA_B_TAU_K;
    gaba_b->reversal = GABA_B_REVERSAL;
    
    // Initialize voltage-gated calcium channel
    ca->g_max = (compartment_type == COMPARTMENT_SOMA) ? CA_G_MAX_SOMA : 
                (compartment_type == COMPARTMENT_SPINE) ? CA_G_MAX_SPINE : CA_G_MAX_DENDRITE;
    ca->reversal = CA_REVERSAL;
    ca->v_half = CA_V_HALF;
    ca->k = CA_K;
    ca->tau_act = CA_TAU_ACT;
    
    // Initialize KCa channel
    kca->g_max = (compartment_type == COMPARTMENT_SOMA) ? KCA_G_MAX_SOMA : KCA_G_MAX_DENDRITE;
    kca->reversal = KCA_REVERSAL;
    kca->ca_half = KCA_CA_HALF;
    kca->hill_coef = KCA_HILL_COEF;
    kca->tau_act = KCA_TAU_ACT;
    
    // Initialize HCN channel
    hcn->g_max = (compartment_type == COMPARTMENT_SOMA) ? HCN_G_MAX_SOMA : HCN_G_MAX_DENDRITE;
    hcn->reversal = HCN_REVERSAL;
    hcn->v_half = HCN_V_HALF;
    hcn->k = HCN_K;
    hcn->tau_min = HCN_TAU_MIN;
    hcn->tau_max = HCN_TAU_MAX;
    hcn->v_tau = HCN_V_TAU;
    hcn->k_tau = HCN_K_TAU;
    
    // Initialize calcium dynamics
    ca_dyn->resting_ca = RESTING_CA_CONCENTRATION;
    ca_dyn->buffer_capacity = CA_BUFFER_CAPACITY;
    ca_dyn->buffer_kd = CA_BUFFER_KD;
    ca_dyn->extrusion_rate = (compartment_type == COMPARTMENT_SOMA) ? 
                             CA_EXTRUSION_RATE_SOMA : CA_EXTRUSION_RATE_DENDRITE;
    ca_dyn->diffusion_rate = CA_DIFFUSION_RATE;
    ca_dyn->volume_factor = (compartment_type == COMPARTMENT_SOMA) ? CA_VOLUME_FACTOR_SOMA :
                            (compartment_type == COMPARTMENT_SPINE) ? CA_VOLUME_FACTOR_SPINE : 
                            CA_VOLUME_FACTOR_DENDRITE;
}