#pragma once
#ifndef NEURON_UPDATE_KERNEL_CUH
#define NEURON_UPDATE_KERNEL_CUH

#include "GPUNeuralStructures.h"
#include <cuda_runtime.h>

/* ------------------------------------------------------------------------- */
/*  Hodgkin–Huxley Runge–Kutta integration                                    */
/* ------------------------------------------------------------------------- */
__global__ void rk4NeuronUpdateKernel(GPUNeuronState* neurons,
                                      int             num_neurons,
                                      float           dt);

/* ------------------------------------------------------------------------- */
/*  Optional passive-leak update for multi-compartment neurons                */
/* ------------------------------------------------------------------------- */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons,
                                     const float*    I_leak,
                                     const float*    Cm,
                                     float           dt,
                                     int             num_neurons);

#endif // NEURON_UPDATE_KERNEL_CUH
