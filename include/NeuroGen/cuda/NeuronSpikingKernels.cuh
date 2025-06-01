#pragma once
#ifndef NEURON_SPIKING_KERNELS_CUH
#define NEURON_SPIKING_KERNELS_CUH

#include <cuda_runtime.h>
#include "../GPUNeuralStructures.h"

/* ------------------------------------------------------------------------- */
/*  Pure “spike bookkeeping” kernels                                         */
/* ------------------------------------------------------------------------- */

/* 1.  Atomic spike counter – returns total spikes in *spike_count            */
__global__ void countSpikesKernel(const GPUNeuronState* neurons,
                                  int                  num_neurons,
                                  int*                 spike_count);

/* 2.  Mark neurons that crossed threshold since last step                    */
__global__ void updateNeuronSpikes(GPUNeuronState* neurons,
                                   int             num_neurons,
                                   float           threshold);

/* 3.  Write spike events into a circular buffer                              */
__global__ void detectSpikes(const GPUNeuronState* neurons,
                             GPUSpikeEvent*        spike_buffer,
                             float                 threshold,
                             int*                  spike_count,
                             int                   num_neurons,
                             float                 current_time);

#endif // NEURON_SPIKING_KERNELS_CUH
