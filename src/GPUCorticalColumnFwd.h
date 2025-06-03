#ifndef GPU_CORTICAL_COLUMN_FWD_H
#define GPU_CORTICAL_COLUMN_FWD_H

// Forward declaration for GPUCorticalColumn for use in non-CUDA compilation
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
    void* d_local_rng_state;    // Device pointer to local RNG state (using void* to avoid CUDA dependency)
    
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

#endif // GPU_CORTICAL_COLUMN_FWD_H
