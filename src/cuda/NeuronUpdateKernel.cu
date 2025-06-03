#include "../../include/NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include "../../include/NeuroGen/GPUNeuralStructures.h"
#include <cuda_runtime.h>
#include <math.h>

// Hodgkin-Huxley model parameters
#define HH_G_NA 120.0f
#define HH_G_K 36.0f
#define HH_G_L 0.3f
#define HH_E_NA 50.0f
#define HH_E_K -77.0f
#define HH_E_L -54.387f

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

// RK4 integration for Hodgkin-Huxley model
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons,
                                     float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState s = neurons[idx];
    
    // Skip inactive neurons
    if (s.active == 0) return;
    
    // Extract state variables
    float v = s.voltage;
    float m = s.m;
    float h = s.h;
    float n = s.n;
    
    // RK4 integration
    // k1
    float I_Na = HH_G_NA * m*m*m * h * (v - HH_E_NA);
    float I_K = HH_G_K * n*n*n*n * (v - HH_E_K);
    float I_L = HH_G_L * (v - HH_E_L);
    float I_total = -(I_Na + I_K + I_L);
    
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
    I_total = -(I_Na + I_K + I_L);
    
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
    I_total = -(I_Na + I_K + I_L);
    
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
    I_total = -(I_Na + I_K + I_L);
    
    float k4_v = dt * I_total;
    float k4_m = dt * (alpha_m(v4) * (1.0f - m4) - beta_m(v4) * m4);
    float k4_h = dt * (alpha_h(v4) * (1.0f - h4) - beta_h(v4) * h4);
    float k4_n = dt * (alpha_n(v4) * (1.0f - n4) - beta_n(v4) * n4);
    
    // Update state variables
    v = v + (k1_v + 2.0f*k2_v + 2.0f*k3_v + k4_v) / 6.0f;
    m = m + (k1_m + 2.0f*k2_m + 2.0f*k3_m + k4_m) / 6.0f;
    h = h + (k1_h + 2.0f*k2_h + 2.0f*k3_h + k4_h) / 6.0f;
    n = n + (k1_n + 2.0f*k2_n + 2.0f*k3_n + k4_n) / 6.0f;
    
    // Clamp values to valid ranges
    if (m < 0.0f) m = 0.0f; else if (m > 1.0f) m = 1.0f;
    if (h < 0.0f) h = 0.0f; else if (h > 1.0f) h = 1.0f;
    if (n < 0.0f) n = 0.0f; else if (n > 1.0f) n = 1.0f;
    
    // Store updated values
    neurons[idx].voltage = v;
    neurons[idx].m = m;
    neurons[idx].h = h;
    neurons[idx].n = n;
    
    // Update compartment voltages
    neurons[idx].voltages[0] = v;
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