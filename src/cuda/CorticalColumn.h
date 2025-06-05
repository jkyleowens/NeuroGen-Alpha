#ifndef GPU_CORTICAL_COLUMN_H
#define GPU_CORTICAL_COLUMN_H

// For CUDA compilation, include the full CUDA version
// For non-CUDA, this file is empty and we use forward declarations from GPUStructuresFwd.h
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * GPU-optimized structure for cortical column organization
 * Only compiled when CUDA is available
 */
struct GPUCorticalColumn {
    // Neuron range within the global neuron array
    int neuron_start;           // Starting index of neurons in this column
    int neuron_end;             // Ending index (exclusive) of neurons in this column
    
    // Synapse range within the global synapse array
    int synapse_start;          // Starting index of synapses originating from this column
    int synapse_end;            // Ending index (exclusive) of synapses from this column
    
    // Column-specific neuromodulation
    float* d_local_dopamine;    // Device pointer to local dopamine concentration
    
    // Column-specific random number generation
    curandState* d_local_rng_state; // Device pointer to local RNG state
    
    // Spatial position and organization
    float center_x, center_y, center_z; // 3D coordinates of column center
    int column_id;              // Unique identifier for this column
    
    // Column properties
    int neuron_count;           // Number of neurons in this column
    int excitatory_count;       // Number of excitatory neurons
    int inhibitory_count;       // Number of inhibitory neurons
    
    // Connectivity statistics
    int local_connections;      // Number of intra-column connections
    int external_connections;   // Number of inter-column connections
    
    // Activity metrics
    float avg_activity;         // Average activity level
    float last_update_time;     // Time of last activity update
    
    // Column state
    bool active;                // Whether this column is currently active
    int developmental_stage;    // Development stage (0=immature, 1=mature)
};

// Utility functions for column management
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a cortical column structure
 */
__host__ __device__ inline void initGPUCorticalColumn(
    GPUCorticalColumn* column,
    int neuron_start,
    int neuron_end,
    int column_id,
    float center_x = 0.0f,
    float center_y = 0.0f,
    float center_z = 0.0f
) {
    column->neuron_start = neuron_start;
    column->neuron_end = neuron_end;
    column->neuron_count = neuron_end - neuron_start;
    column->column_id = column_id;
    column->center_x = center_x;
    column->center_y = center_y;
    column->center_z = center_z;
    
    // Initialize other fields to default values
    column->synapse_start = 0;
    column->synapse_end = 0;
    column->d_local_dopamine = nullptr;
    column->d_local_rng_state = nullptr;
    column->excitatory_count = 0;
    column->inhibitory_count = 0;
    column->local_connections = 0;
    column->external_connections = 0;
    column->avg_activity = 0.0f;
    column->last_update_time = 0.0f;
    column->active = true;
    column->developmental_stage = 1; // Default to mature
}

/**
 * Check if a neuron index belongs to this column
 */
__host__ __device__ inline bool columnContainsNeuron(
    const GPUCorticalColumn* column,
    int neuron_idx
) {
    return neuron_idx >= column->neuron_start && neuron_idx < column->neuron_end;
}

/**
 * Get the relative neuron index within the column
 */
__host__ __device__ inline int getRelativeNeuronIndex(
    const GPUCorticalColumn* column,
    int global_neuron_idx
) {
    return global_neuron_idx - column->neuron_start;
}

}

#endif // __CUDACC__

#endif // GPU_CORTICAL_COLUMN_H
