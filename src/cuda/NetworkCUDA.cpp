#include "NetworkCUDA.cuh"
#include <vector>

void NetworkCUDA::initializeNetwork()
{
    /* ------------------------------------------------------------------ */
    /* 0.  Guard: don’t initialise twice                                   */
    /* ------------------------------------------------------------------ */
    if (d_neurons || d_synapses || d_columns)
        throw std::runtime_error("NetworkCUDA already initialised");

    /* ------------------------------------------------------------------ */
    /* 1.  Compute global sizes                                            */
    /* ------------------------------------------------------------------ */
    total_neurons  = config.numColumns * config.neuronsPerColumn;
    total_synapses = static_cast<int>(config.numColumns) *
                     config.neuronsPerColumn *
                     config.localFanOut;          // fan-out = #outgoing / cell

    /* ------------------------------------------------------------------ */
    /* 2.  Allocate the *flat* neuron & synapse device arrays              */
    /* ------------------------------------------------------------------ */
    allocateDeviceMemory(&d_neurons,  total_neurons);
    allocateDeviceMemory(&d_synapses, total_synapses);

    /* ------------------------------------------------------------------ */
    /* 3.  Build host-side column descriptors                              */
    /* ------------------------------------------------------------------ */
    h_columns.clear();
    h_columns.reserve(config.numColumns);

    int neuron_cursor  = 0;
    int synapse_cursor = 0;
    for (int c = 0; c < config.numColumns; ++c)
    {
        GPUCorticalColumn col{};

        /* neuron slice [start, end) ------------------------------------ */
        col.neuron_start = neuron_cursor;
        col.neuron_end   = neuron_cursor + config.neuronsPerColumn;
        neuron_cursor    = col.neuron_end;

        /* synapse slice filled *after* we build the topology ------------ */
        /* leave placeholders for now; we’ll patch them in a minute       */
        col.synapse_start      = 0;
        col.synapse_end        = 0;
        col.d_local_dopamine   = nullptr;
        col.d_local_rng_state  = nullptr;

        /* optional per-column buffers (1 value each for now) ------------ */
        allocateDeviceMemory(&col.d_local_dopamine,  1);
        allocateDeviceMemory(&col.d_local_rng_state, 1);

        h_columns.push_back(col);
    }
    n_columns = static_cast<int>(h_columns.size());

    /* ------------------------------------------------------------------ */
    /* 4.  Generate local recurrent synapses (host side)                   */
    /* ------------------------------------------------------------------ */
    std::vector<GPUSynapse> host_synapses;
    host_synapses.reserve(total_synapses);

    TopologyGenerator topo(config);
    topo.buildLocalLoops(host_synapses, h_columns);      // fills vector

    /*  Patch column.synapse_* now that we know exact counts ------------- */
    synapse_cursor = 0;
    for (auto& col : h_columns)
    {
        col.synapse_start = synapse_cursor;
        col.synapse_end   = synapse_cursor +
                            (config.neuronsPerColumn * config.localFanOut);
        synapse_cursor    = col.synapse_end;
    }

    /* ------------------------------------------------------------------ */
    /* 5.  Copy synapses + columns to device                               */
    /* ------------------------------------------------------------------ */
    CUDA_CHECK(cudaMemcpy(d_synapses,
                          host_synapses.data(),
                          host_synapses.size() * sizeof(GPUSynapse),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_columns,
                          h_columns.size() * sizeof(GPUCorticalColumn)));
    CUDA_CHECK(cudaMemcpy(d_columns,
                          h_columns.data(),
                          h_columns.size() * sizeof(GPUCorticalColumn),
                          cudaMemcpyHostToDevice));

    /* ------------------------------------------------------------------ */
    /* 6.  RNG pool for spike-timing jitters, structural plasticity, etc.  */
    /* ------------------------------------------------------------------ */
    allocateDeviceMemory(&d_rng_states, total_neurons);

    dim3 grid ((total_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    initRandomStates<<<grid, block>>>(d_rng_states, total_neurons);
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal) {
    // Perform a forward pass through the CUDA network
    // Placeholder logic: return a dummy output vector
    return {0.1f, 0.2f, 0.7f};
}

void updateSynapticWeightsCUDA(float reward_signal) {
    // Perform STDP updates on synaptic weights using the reward signal
    // Placeholder logic: no-op
}

