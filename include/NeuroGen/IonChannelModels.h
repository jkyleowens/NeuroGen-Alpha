#ifndef ION_CHANNEL_MODELS_H
#define ION_CHANNEL_MODELS_H

#include <cuda_runtime.h>
#include <NeuroGen/cuda/NeuronModelConstants.h>

/**
 * AMPA receptor model - Fast excitatory neurotransmitter receptor
 * Mediates rapid depolarization with simple dual-exponential kinetics
 */
struct AMPAChannel {
    float g_max;        // Maximum conductance (nS)
    float tau_rise;     // Rise time constant (ms)
    float tau_decay;    // Decay time constant (ms)  
    float reversal;     // Reversal potential (mV)
    
    __device__ float computeCurrent(float v, float g) {
        return g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Dual exponential synapse model
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
        
        // Prevent negative conductances
        if (g < 0.0f) g = 0.0f;
        if (state < 0.0f) state = 0.0f;
    }
};

/**
 * NMDA receptor model with voltage-dependent Mg2+ block
 * Critical for spike-timing dependent plasticity and calcium influx
 */
struct NMDAChannel {
    float g_max;        // Maximum conductance (nS)
    float tau_rise;     // Rise time constant (ms)
    float tau_decay;    // Decay time constant (ms)
    float reversal;     // Reversal potential (mV)
    float mg_conc;      // Magnesium concentration (mM)
    float ca_fraction;  // Fraction of current carried by Ca2+
    
    __device__ float computeMgBlock(float v) {
        // Voltage-dependent magnesium block (Jahr & Stevens, 1990)
        return 1.0f / (1.0f + (mg_conc / 3.57f) * expf(-0.062f * v));
    }
    
    __device__ float computeCurrent(float v, float g) {
        float mg_block = computeMgBlock(v);
        return g * mg_block * (v - reversal);
    }
    
    __device__ float computeCalciumCurrent(float v, float g) {
        float mg_block = computeMgBlock(v);
        float total_current = g * mg_block * (v - reversal);
        return ca_fraction * total_current;
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Dual exponential with slower kinetics than AMPA
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
        
        if (g < 0.0f) g = 0.0f;
        if (state < 0.0f) state = 0.0f;
    }
};

/**
 * GABA-A receptor model - Fast inhibitory neurotransmitter receptor
 * Provides rapid hyperpolarization through chloride channels
 */
struct GABAA_Channel {
    float g_max;        // Maximum conductance (nS)
    float tau_rise;     // Rise time constant (ms)
    float tau_decay;    // Decay time constant (ms)
    float reversal;     // Reversal potential (mV) - typically -70mV
    
    __device__ float computeCurrent(float v, float g) {
        return g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float input, float dt) {
        // Fast inhibitory kinetics
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
        
        if (g < 0.0f) g = 0.0f;
        if (state < 0.0f) state = 0.0f;
    }
};

/**
 * GABA-B receptor model - Slow inhibitory receptor with G-protein coupling
 * Provides prolonged hyperpolarization through potassium channels
 */
struct GABAB_Channel {
    float g_max;        // Maximum conductance (nS)
    float tau_rise;     // Rise time constant (ms)
    float tau_decay;    // Decay time constant (ms)
    float tau_k;        // K+ channel activation time constant (ms)
    float reversal;     // Reversal potential (mV) - typically -90mV
    
    __device__ float computeCurrent(float v, float g, float g_protein) {
        // Current depends on both conductance and G-protein activation
        float effective_g = g * g_protein;
        return effective_g * (v - reversal);
    }
    
    __device__ void updateState(float& g, float& state, float& g_protein, float input, float dt) {
        // Receptor activation
        float dgdt = -g / tau_decay + state;
        float dstate_dt = -state / tau_rise + input;
        
        // G-protein cascade activation
        float dg_protein_dt = (g - g_protein) / tau_k;
        
        g += dgdt * dt;
        state += dstate_dt * dt;
        g_protein += dg_protein_dt * dt;
        
        if (g < 0.0f) g = 0.0f;
        if (state < 0.0f) state = 0.0f;
        if (g_protein < 0.0f) g_protein = 0.0f;
        if (g_protein > 1.0f) g_protein = 1.0f;
    }
};

/**
 * Voltage-gated calcium channel (L-type)
 * Critical for calcium influx and activity-dependent processes
 */
struct CaChannel {
    float g_max;        // Maximum conductance (nS)
    float reversal;     // Reversal potential (mV)
    float v_half;       // Half-activation voltage (mV)
    float k;            // Slope factor (mV)
    float tau_act;      // Activation time constant (ms)
    
    __device__ float steadyStateActivation(float v) {
        return 1.0f / (1.0f + expf(-(v - v_half) / k));
    }
    
    __device__ float computeCurrent(float v, float m) {
        return g_max * m * (v - reversal);
    }
    
    __device__ void updateState(float& m, float v, float dt) {
        float m_inf = steadyStateActivation(v);
        float dmdt = (m_inf - m) / tau_act;
        m += dmdt * dt;
        
        if (m < 0.0f) m = 0.0f;
        if (m > 1.0f) m = 1.0f;
    }
};

/**
 * Calcium-dependent potassium channel (KCa)
 * Provides calcium-dependent after-hyperpolarization
 */
struct KCaChannel {
    float g_max;        // Maximum conductance (nS)
    float reversal;     // Reversal potential (mV)
    float ca_half;      // Half-activation calcium concentration (mM)
    float hill_coef;    // Hill coefficient
    float tau_act;      // Activation time constant (ms)
    
    __device__ float calciumDependentActivation(float ca_conc) {
        float ca_term = powf(ca_conc, hill_coef);
        float ca_half_term = powf(ca_half, hill_coef);
        return ca_term / (ca_term + ca_half_term);
    }
    
    __device__ float computeCurrent(float v, float m) {
        return g_max * m * (v - reversal);
    }
    
    __device__ void updateState(float& m, float ca_conc, float dt) {
        float m_inf = calciumDependentActivation(ca_conc);
        float dmdt = (m_inf - m) / tau_act;
        m += dmdt * dt;
        
        if (m < 0.0f) m = 0.0f;
        if (m > 1.0f) m = 1.0f;
    }
};

/**
 * Hyperpolarization-activated cation channel (HCN/Ih)
 * Provides membrane resonance and pacemaker activity
 */
struct HCNChannel {
    float g_max;        // Maximum conductance (nS)
    float reversal;     // Reversal potential (mV)
    float v_half;       // Half-activation voltage (mV)
    float k;            // Slope factor (mV)
    float tau_min;      // Minimum time constant (ms)
    float tau_max;      // Maximum time constant (ms)
    float v_tau;        // Voltage for tau calculation (mV)
    float k_tau;        // Slope for tau calculation (mV)
    
    __device__ float steadyStateActivation(float v) {
        return 1.0f / (1.0f + expf((v - v_half) / k));  // Note: positive slope
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
        
        if (h < 0.0f) h = 0.0f;
        if (h > 1.0f) h = 1.0f;
    }
};

/**
 * Calcium dynamics structure
 * Handles calcium buffering, diffusion, and extrusion
 */
struct CalciumDynamics {
    float resting_ca;       // Resting calcium concentration (mM)
    float buffer_capacity;  // Buffer capacity (relative units)
    float buffer_kd;        // Buffer dissociation constant (mM)
    float extrusion_rate;   // Calcium pump rate (1/ms)
    float diffusion_rate;   // Diffusion coefficient (1/ms)
    float volume_factor;    // Conversion from current to concentration
    
    __device__ float computeBuffering(float ca_conc, float buffer_conc) {
        // Hill equation for calcium buffering
        return buffer_capacity * buffer_conc / (buffer_kd + ca_conc);
    }
    
    __device__ void updateCalcium(float& ca_conc, float& buffer_conc, 
                                 float i_ca, float dt) {
        // Calcium influx from current (negative current increases calcium)
        float ca_influx = -i_ca * volume_factor;
        
        // Calcium buffering
        float buffer_binding = computeBuffering(ca_conc, buffer_conc);
        
        // Calcium extrusion
        float ca_extrusion = extrusion_rate * (ca_conc - resting_ca);
        
        // Update calcium concentration
        float dca_dt = ca_influx - ca_extrusion - buffer_binding;
        ca_conc += dca_dt * dt;
        
        // Update buffer concentration
        float dbuffer_dt = buffer_binding;
        buffer_conc += dbuffer_dt * dt;
        
        // Enforce bounds
        if (ca_conc < resting_ca) ca_conc = resting_ca;
        if (buffer_conc < 0.0f) buffer_conc = 0.0f;
    }
};

#endif // ION_CHANNEL_MODELS_H