// NeuronUpdateKernel.cu – full HH + RK4 implementation
#include "../../include/NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>

/* HH constants (mV, mS/cm²) ------------------------------------------------ */
#define ENa   50.0f
#define EK   -77.0f
#define EL   -54.387f
#define gNa 120.0f
#define gK   36.0f
#define gL    0.3f
#define EPS  1e-6f

/* Small helper to avoid /0 -------------------------------------------------- */
__device__ inline float safe_den(float x) { return fabsf(x) < EPS ? EPS : x; }

/* Channel rate equations ---------------------------------------------------- */
__device__ float alpha_n(float V) { return (0.01f * (V + 55.0f)) / safe_den(1.0f - expf(-0.1f * (V + 55.0f))); }
__device__ float beta_n (float V) { return 0.125f * expf(-(V + 65.0f) / 80.0f); }

__device__ float alpha_m(float V) { return (0.1f  * (V + 40.0f)) / safe_den(1.0f - expf(-0.1f * (V + 40.0f))); }
__device__ float beta_m (float V) { return 4.0f   * expf(-(V + 65.0f) / 18.0f); }

__device__ float alpha_h(float V) { return 0.07f  * expf(-(V + 65.0f) / 20.0f); }
__device__ float beta_h (float V) { return 1.0f   / safe_den(1.0f + expf(-0.1f * (V + 35.0f))); }

/* dV/dt --------------------------------------------------------------------- */
__device__ float dVdt(float V, float m, float h, float n)
{
    float INa = gNa * powf(m, 3) * h  * (V - ENa);
    float IK  = gK  * powf(n, 4)       * (V - EK );
    float IL  = gL                     * (V - EL );
    return -(INa + IK + IL);           // Cm = 1 µF/cm²
}

/* Main RK4 kernel ----------------------------------------------------------- */
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons,
                                      int             num_neurons,
                                      float           dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_neurons) return;

    /* Load state */
    GPUNeuronState s = neurons[idx];
    float V = s.voltage, m = s.m, h = s.h, n = s.n;

    /* ---- stage 1 ---- */
    float k1  = dVdt(V, m, h, n);
    float dm1 = alpha_m(V)*(1 - m) - beta_m(V)*m;
    float dh1 = alpha_h(V)*(1 - h) - beta_h(V)*h;
    float dn1 = alpha_n(V)*(1 - n) - beta_n(V)*n;

    /* ---- stage 2 ---- */
    float V2  = V + 0.5f*dt*k1;
    float m2  = m + 0.5f*dt*dm1;
    float h2  = h + 0.5f*dt*dh1;
    float n2  = n + 0.5f*dt*dn1;
    float k2  = dVdt(V2, m2, h2, n2);
    float dm2 = alpha_m(V2)*(1 - m2) - beta_m(V2)*m2;
    float dh2 = alpha_h(V2)*(1 - h2) - beta_h(V2)*h2;
    float dn2 = alpha_n(V2)*(1 - n2) - beta_n(V2)*n2;

    /* ---- stage 3 ---- */
    float V3  = V + 0.5f*dt*k2;
    float m3  = m + 0.5f*dt*dm2;
    float h3  = h + 0.5f*dt*dh2;
    float n3  = n + 0.5f*dt*dn2;
    float k3  = dVdt(V3, m3, h3, n3);
    float dm3 = alpha_m(V3)*(1 - m3) - beta_m(V3)*m3;
    float dh3 = alpha_h(V3)*(1 - h3) - beta_h(V3)*h3;
    float dn3 = alpha_n(V3)*(1 - n3) - beta_n(V3)*n3;

    /* ---- stage 4 ---- */
    float V4  = V + dt*k3;
    float m4  = m + dt*dm3;
    float h4  = h + dt*dh3;
    float n4  = n + dt*dn3;
    float k4  = dVdt(V4, m4, h4, n4);
    float dm4 = alpha_m(V4)*(1 - m4) - beta_m(V4)*m4;
    float dh4 = alpha_h(V4)*(1 - h4) - beta_h(V4)*h4;
    float dn4 = alpha_n(V4)*(1 - n4) - beta_n(V4)*n4;

    /* ---- combine ---- */
    V += (dt / 6.0f) * (k1  + 2*k2  + 2*k3  + k4 );
    m += (dt / 6.0f) * (dm1 + 2*dm2 + 2*dm3 + dm4);
    h += (dt / 6.0f) * (dh1 + 2*dh2 + 2*dh3 + dh4);
    n += (dt / 6.0f) * (dn1 + 2*dn2 + 2*dn3 + dn4);

    /* Write back */
    if (!isnan(V) && !isinf(V)) {
        s.voltage = V;  s.m = m;  s.h = h;  s.n = n;
        neurons[idx] = s;
    }
}

/* Optional passive leak for multi-compartment models ----------------------- */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons,
                                     const float*    I_leak,
                                     const float*    Cm,
                                     float           dt,
                                     int             num_neurons)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_neurons) return;

    neurons[idx].voltage += dt * (-I_leak[idx]) / Cm[idx];
}
