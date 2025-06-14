// PerformanceOptimization.h
#ifndef PERFORMANCE_OPTIMIZATION_H
#define PERFORMANCE_OPTIMIZATION_H

#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <vector>

/**
 * Performance optimization utilities for ion channel dynamics
 * Includes memory coalescing, shared memory usage, and computation reduction
 */

// ========================================
// MEMORY-OPTIMIZED NEURON STRUCTURE
// ========================================

/**
 * Structure of Arrays (SoA) layout for better memory coalescing
 * Separates frequently accessed data from infrequently accessed data
 */
struct OptimizedNeuronArrays {
    // Hot data - accessed every timestep
    float* voltages;                    // [neuron_id * MAX_COMPARTMENTS + comp_id]
    float* ca_concentrations;           // [neuron_id * MAX_COMPARTMENTS + comp_id]
    
    // Channel states - frequently accessed
    float* ampa_conductances;           // [neuron_id * MAX_COMPARTMENTS + comp_id]
    float* nmda_conductances;
    float* gaba_a_conductances;
    float* gaba_b_conductances;
    float* ca_channel_states;
    float* kca_channel_states;
    float* hcn_channel_states;
    
    // Hodgkin-Huxley states
    float* hh_m_states;
    float* hh_h_states;
    float* hh_n_states;
    
    // Cold data - accessed less frequently
    int* neuron_active_flags;
    int* compartment_types;
    int* compartment_counts;
    float* coupling_conductances;
    
    // Metadata
    int total_neurons;
    int max_compartments;
    size_t total_elements;
};

/**
 * Optimized synapse structure for better cache performance
 */
struct OptimizedSynapseArrays {
    // Critical path data
    int* pre_neuron_indices;
    int* post_neuron_indices;
    int* post_compartment_indices;
    int* receptor_types;
    float* weights;
    float* effective_weights;
    
    // Activity data
    float* last_spike_times;
    float* activity_metrics;
    int* vesicle_counts;
    float* release_probabilities;
    
    // Plasticity data (accessed less frequently)
    float* eligibility_traces;
    float* plasticity_rates;
    
    int total_synapses;
};

/**
 * Performance monitoring structure
 */
struct PerformanceMetrics {
    double neuron_update_time_ms;
    double synapse_processing_time_ms;
    double calcium_dynamics_time_ms;
    double memory_transfer_time_ms;
    double total_simulation_time_ms;
    
    size_t peak_memory_usage_bytes;
    float memory_bandwidth_gb_s;
    float computational_throughput_gflops;
    
    int neurons_processed_per_ms;
    int synapses_processed_per_ms;
};

// ========================================
// CUDA KERNEL OPTIMIZATIONS
// ========================================

/**
 * Optimized neuron update kernel using shared memory and reduced branching
 */
__global__ void optimizedNeuronUpdateKernel(
    OptimizedNeuronArrays arrays,
    float dt,
    float current_time,
    int batch_size
) {
    // Use shared memory for frequently accessed constants
    __shared__ float shared_constants[32];
    
    if (threadIdx.x < 32) {
        // Load constants into shared memory
        shared_constants[0] = HH_G_NA;
        shared_constants[1] = HH_G_K;
        shared_constants[2] = HH_G_L;
        shared_constants[3] = HH_E_NA;
        shared_constants[4] = HH_E_K;
        shared_constants[5] = HH_E_L;
        shared_constants[6] = AMPA_REVERSAL;
        shared_constants[7] = NMDA_REVERSAL;
        shared_constants[8] = GABA_A_REVERSAL;
        shared_constants[9] = GABA_B_REVERSAL;
        // ... load more constants
    }
    __syncthreads();
    
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= arrays.total_neurons) return;
    
    // Check if neuron is active (early exit for inactive neurons)
    if (arrays.neuron_active_flags[neuron_idx] == 0) return;
    
    int compartment_count = arrays.compartment_counts[neuron_idx];
    
    // Process each compartment with reduced memory accesses
    for (int c = 0; c < compartment_count; c++) {
        int global_idx = neuron_idx * arrays.max_compartments + c;
        
        // Load compartment type once and use for all decisions
        int comp_type = arrays.compartment_types[global_idx];
        if (comp_type == COMPARTMENT_INACTIVE) continue;
        
        // Load current states in bulk
        float v = arrays.voltages[global_idx];
        float ca = arrays.ca_concentrations[global_idx];
        float m = arrays.hh_m_states[global_idx];
        float h = arrays.hh_h_states[global_idx];
        float n = arrays.hh_n_states[global_idx];
        
        // Load conductances
        float ampa_g = arrays.ampa_conductances[global_idx];
        float nmda_g = arrays.nmda_conductances[global_idx];
        float gaba_a_g = arrays.gaba_a_conductances[global_idx];
        float gaba_b_g = arrays.gaba_b_conductances[global_idx];
        
        // Compute currents using shared constants
        float I_Na = shared_constants[0] * m*m*m * h * (v - shared_constants[3]);
        float I_K = shared_constants[1] * n*n*n*n * (v - shared_constants[4]);
        float I_L = shared_constants[2] * (v - shared_constants[5]);
        
        // Synaptic currents with reduced NMDA Mg block calculation
        float I_AMPA = ampa_g * (v - shared_constants[6]);
        
        // Optimized NMDA Mg block (use lookup table or approximation)
        float mg_factor = 1.0f / (1.0f + 0.28f * __expf(-0.062f * v));
        float I_NMDA = nmda_g * mg_factor * (v - shared_constants[7]);
        
        float I_GABA_A = gaba_a_g * (v - shared_constants[8]);
        float I_GABA_B = gaba_b_g * (v - shared_constants[9]);
        
        // Total current
        float I_total = -(I_Na + I_K + I_L + I_AMPA + I_NMDA + I_GABA_A + I_GABA_B);
        
        // Simplified RK4 integration (or use Euler for speed)
        float dv_dt = I_total / 100.0f;  // Assume 100 pF capacitance
        v += dv_dt * dt;
        
        // Update HH variables with fast approximations
        float alpha_m_v = (0.1f * (v + 40.0f)) / (1.0f - __expf(-0.1f * (v + 40.0f)));
        float beta_m_v = 4.0f * __expf(-(v + 65.0f) / 18.0f);
        float m_inf = alpha_m_v / (alpha_m_v + beta_m_v);
        
        m += (m_inf - m) * dt / 1.0f;  // Fast time constant
        
        // Similar for h and n (simplified)
        h += ((1.0f / (1.0f + __expf((v + 35.0f) / 10.0f))) - h) * dt / 5.0f;
        n += ((1.0f / (1.0f + __expf(-(v + 55.0f) / 10.0f))) - n) * dt / 10.0f;
        
        // Clamp values
        v = fmaxf(-100.0f, fminf(50.0f, v));
        m = fmaxf(0.0f, fminf(1.0f, m));
        h = fmaxf(0.0f, fminf(1.0f, h));
        n = fmaxf(0.0f, fminf(1.0f, n));
        
        // Store results (coalesced writes)
        arrays.voltages[global_idx] = v;
        arrays.hh_m_states[global_idx] = m;
        arrays.hh_h_states[global_idx] = h;
        arrays.hh_n_states[global_idx] = n;
    }
}

/**
 * Optimized calcium dynamics kernel with reduced precision for speed
 */
__global__ void optimizedCalciumKernel(
    OptimizedNeuronArrays arrays,
    float dt,
    int batch_size
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= arrays.total_neurons) return;
    
    if (arrays.neuron_active_flags[neuron_idx] == 0) return;
    
    int compartment_count = arrays.compartment_counts[neuron_idx];
    
    // Use registers for temporary storage
    float ca_values[MAX_COMPARTMENTS];
    
    // Load calcium values
    for (int c = 0; c < compartment_count; c++) {
        int global_idx = neuron_idx * arrays.max_compartments + c;
        ca_values[c] = arrays.ca_concentrations[global_idx];
    }
    
    // Process calcium dynamics with simplified model
    for (int c = 0; c < compartment_count; c++) {
        int global_idx = neuron_idx * arrays.max_compartments + c;
        
        if (arrays.compartment_types[global_idx] == COMPARTMENT_INACTIVE) continue;
        
        float ca = ca_values[c];
        
        // Simplified calcium extrusion (linear model)
        float extrusion = 0.1f * (ca - RESTING_CA_CONCENTRATION) * dt;
        
        // Simple diffusion to neighboring compartments
        float diffusion = 0.0f;
        if (c > 0) {  // Not soma
            diffusion += 0.01f * (ca_values[0] - ca) * dt;  // Diffusion to soma
        }
        if (c == 0 && compartment_count > 1) {  // Soma
            for (int child = 1; child < compartment_count; child++) {
                diffusion += 0.01f * (ca_values[child] - ca) * dt;
            }
        }
        
        // Update calcium
        ca = ca - extrusion + diffusion;
        ca = fmaxf(MIN_CA_CONCENTRATION, fminf(MAX_CA_CONCENTRATION, ca));
        
        arrays.ca_concentrations[global_idx] = ca;
    }
}

/**
 * Batch processing utilities for improved throughput
 */
class BatchProcessor {
public:
    static void processBatchedNeuronUpdate(
        OptimizedNeuronArrays& arrays,
        float dt,
        float current_time,
        int batch_size = 1024
    ) {
        int num_batches = (arrays.total_neurons + batch_size - 1) / batch_size;
        
        dim3 block(min(256, batch_size));
        dim3 grid(num_batches);
        
        // Use CUDA streams for overlapping computation
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
        // Process batches with overlapping
        for (int batch = 0; batch < num_batches; batch += 2) {
            // Launch first batch
            optimizedNeuronUpdateKernel<<<grid, block, 0, stream1>>>(
                arrays, dt, current_time, batch_size
            );
            
            // Launch second batch if available
            if (batch + 1 < num_batches) {
                optimizedNeuronUpdateKernel<<<grid, block, 0, stream2>>>(
                    arrays, dt, current_time, batch_size
                );
            }
            
            // Synchronize streams
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
        }
        
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
};

// ========================================
// MEMORY MANAGEMENT OPTIMIZATIONS
// ========================================

/**
 * Memory pool allocator for reducing allocation overhead
 */
class CudaMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<MemoryBlock> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
    
public:
    CudaMemoryPool() : total_allocated_(0), peak_usage_(0) {}
    
    ~CudaMemoryPool() {
        cleanup();
    }
    
    void* allocate(size_t size) {
        // Find suitable block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        blocks_.push_back({ptr, size, true});
        total_allocated_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    void cleanup() {
        for (const auto& block : blocks_) {
            cudaFree(block.ptr);
        }
        blocks_.clear();
        total_allocated_ = 0;
    }
    
    size_t getPeakUsage() const { return peak_usage_; }
    size_t getCurrentUsage() const { return total_allocated_; }
};

// ========================================
// PERFORMANCE PROFILING
// ========================================

/**
 * Performance profiler for ion channel simulations
 */
class IonChannelProfiler {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    PerformanceMetrics metrics_;
    bool profiling_active_;
    
public:
    IonChannelProfiler() : profiling_active_(false) {
        memset(&metrics_, 0, sizeof(PerformanceMetrics));
    }
    
    void startProfiling() {
        profiling_active_ = true;
        start_time_ = std::chrono::high_resolution_clock::now();
        cudaProfilerStart();
    }
    
    void stopProfiling() {
        if (!profiling_active_) return;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        metrics_.total_simulation_time_ms = duration.count() / 1000.0;
        
        cudaProfilerStop();
        profiling_active_ = false;
    }
    
    void recordNeuronUpdateTime(double time_ms) {
        metrics_.neuron_update_time_ms += time_ms;
    }
    
    void recordSynapseProcessingTime(double time_ms) {
        metrics_.synapse_processing_time_ms += time_ms;
    }
    
    void recordCalciumDynamicsTime(double time_ms) {
        metrics_.calcium_dynamics_time_ms += time_ms;
    }
    
    void recordMemoryTransferTime(double time_ms) {
        metrics_.memory_transfer_time_ms += time_ms;
    }
    
    void calculateThroughput(int neurons_processed, int synapses_processed) {
        if (metrics_.total_simulation_time_ms > 0) {
            metrics_.neurons_processed_per_ms = neurons_processed / metrics_.total_simulation_time_ms;
            metrics_.synapses_processed_per_ms = synapses_processed / metrics_.total_simulation_time_ms;
        }
    }
    
    const PerformanceMetrics& getMetrics() const {
        return metrics_;
    }
    
    void printPerformanceReport() {
        printf("\n=== Ion Channel Performance Report ===\n");
        printf("Total simulation time: %.2f ms\n", metrics_.total_simulation_time_ms);
        printf("Neuron update time: %.2f ms (%.1f%%)\n", 
               metrics_.neuron_update_time_ms,
               100.0 * metrics_.neuron_update_time_ms / metrics_.total_simulation_time_ms);
        printf("Synapse processing time: %.2f ms (%.1f%%)\n", 
               metrics_.synapse_processing_time_ms,
               100.0 * metrics_.synapse_processing_time_ms / metrics_.total_simulation_time_ms);
        printf("Calcium dynamics time: %.2f ms (%.1f%%)\n", 
               metrics_.calcium_dynamics_time_ms,
               100.0 * metrics_.calcium_dynamics_time_ms / metrics_.total_simulation_time_ms);
        printf("Memory transfer time: %.2f ms (%.1f%%)\n", 
               metrics_.memory_transfer_time_ms,
               100.0 * metrics_.memory_transfer_time_ms / metrics_.total_simulation_time_ms);
        printf("\nThroughput:\n");
        printf("Neurons processed: %d/ms\n", metrics_.neurons_processed_per_ms);
        printf("Synapses processed: %d/ms\n", metrics_.synapses_processed_per_ms);
        printf("Peak memory usage: %.2f MB\n", metrics_.peak_memory_usage_bytes / (1024.0 * 1024.0));
        printf("======================================\n\n");
    }
};

// ========================================
// OPTIMIZATION UTILITIES
// ========================================

/**
 * Utility functions for performance optimization
 */
class OptimizationUtils {
public:
    /**
     * Convert AoS to SoA layout for better memory coalescing
     */
    static bool convertToOptimizedLayout(
        GPUNeuronState* d_neurons,
        int num_neurons,
        OptimizedNeuronArrays& optimized_arrays
    ) {
        // Calculate total elements
        optimized_arrays.total_neurons = num_neurons;
        optimized_arrays.max_compartments = MAX_COMPARTMENTS;
        optimized_arrays.total_elements = num_neurons * MAX_COMPARTMENTS;
        
        // Allocate optimized arrays
        CudaMemoryPool& pool = getCudaMemoryPool();
        
        size_t float_array_size = optimized_arrays.total_elements * sizeof(float);
        size_t int_array_size = optimized_arrays.total_elements * sizeof(int);
        
        optimized_arrays.voltages = (float*)pool.allocate(float_array_size);
        optimized_arrays.ca_concentrations = (float*)pool.allocate(float_array_size);
        optimized_arrays.ampa_conductances = (float*)pool.allocate(float_array_size);
        optimized_arrays.nmda_conductances = (float*)pool.allocate(float_array_size);
        optimized_arrays.gaba_a_conductances = (float*)pool.allocate(float_array_size);
        optimized_arrays.gaba_b_conductances = (float*)pool.allocate(float_array_size);
        optimized_arrays.ca_channel_states = (float*)pool.allocate(float_array_size);
        optimized_arrays.kca_channel_states = (float*)pool.allocate(float_array_size);
        optimized_arrays.hcn_channel_states = (float*)pool.allocate(float_array_size);
        optimized_arrays.hh_m_states = (float*)pool.allocate(float_array_size);
        optimized_arrays.hh_h_states = (float*)pool.allocate(float_array_size);
        optimized_arrays.hh_n_states = (float*)pool.allocate(float_array_size);
        
        optimized_arrays.neuron_active_flags = (int*)pool.allocate(num_neurons * sizeof(int));
        optimized_arrays.compartment_types = (int*)pool.allocate(int_array_size);
        optimized_arrays.compartment_counts = (int*)pool.allocate(num_neurons * sizeof(int));
        optimized_arrays.coupling_conductances = (float*)pool.allocate(float_array_size);
        
        // Launch conversion kernel
        dim3 block(256);
        dim3 grid((num_neurons + block.x - 1) / block.x);
        
        convertAoSToSoAKernel<<<grid, block>>>(d_neurons, optimized_arrays, num_neurons);
        
        cudaError_t err = cudaGetLastError();
        return err == cudaSuccess;
    }
    
    /**
     * Auto-tune kernel launch parameters
     */
    static dim3 getOptimalBlockSize(int num_elements, int max_threads_per_block = 1024) {
        int threads = min(max_threads_per_block, num_elements);
        
        // Prefer multiples of warp size (32)
        threads = (threads / 32) * 32;
        if (threads == 0) threads = 32;
        
        return dim3(threads);
    }
    
    static dim3 getOptimalGridSize(int num_elements, int block_size) {
        return dim3((num_elements + block_size - 1) / block_size);
    }
    
    /**
     * Memory bandwidth test
     */
    static float measureMemoryBandwidth(size_t array_size_bytes) {
        float* d_src;
        float* d_dst;
        
        cudaMalloc(&d_src, array_size_bytes);
        cudaMalloc(&d_dst, array_size_bytes);
        
        // Warm up
        cudaMemcpy(d_dst, d_src, array_size_bytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        
        // Measure bandwidth
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10; i++) {
            cudaMemcpy(d_dst, d_src, array_size_bytes, cudaMemcpyDeviceToDevice);
        }
        
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        float time_seconds = duration.count() / 1000000.0f;
        float bandwidth_gb_s = (array_size_bytes * 10 * 2) / (time_seconds * 1e9);  // Read + Write
        
        cudaFree(d_src);
        cudaFree(d_dst);
        
        return bandwidth_gb_s;
    }
    
private:
    static CudaMemoryPool& getCudaMemoryPool() {
        static CudaMemoryPool pool;
        return pool;
    }
};

/**
 * Kernel for converting AoS to SoA layout
 */
__global__ void convertAoSToSoAKernel(
    GPUNeuronState* aos_neurons,
    OptimizedNeuronArrays soa_arrays,
    int num_neurons
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    GPUNeuronState& neuron = aos_neurons[neuron_idx];
    
    // Copy neuron-level data
    soa_arrays.neuron_active_flags[neuron_idx] = neuron.active;
    soa_arrays.compartment_counts[neuron_idx] = neuron.compartment_count;
    
    // Copy compartment data
    for (int c = 0; c < neuron.compartment_count; c++) {
        int global_idx = neuron_idx * soa_arrays.max_compartments + c;
        
        soa_arrays.voltages[global_idx] = (c == 0) ? neuron.voltage : neuron.voltages[c];
        soa_arrays.ca_concentrations[global_idx] = neuron.ca_conc[c];
        soa_arrays.compartment_types[global_idx] = neuron.compartment_types[c];
        soa_arrays.coupling_conductances[global_idx] = neuron.coupling_conductance[c];
        
        // Channel states
        soa_arrays.ampa_conductances[global_idx] = neuron.channels.ampa_g[c];
        soa_arrays.nmda_conductances[global_idx] = neuron.channels.nmda_g[c];
        soa_arrays.gaba_a_conductances[global_idx] = neuron.channels.gaba_a_g[c];
        soa_arrays.gaba_b_conductances[global_idx] = neuron.channels.gaba_b_g[c];
        soa_arrays.ca_channel_states[global_idx] = neuron.channels.ca_m[c];
        soa_arrays.kca_channel_states[global_idx] = neuron.channels.kca_m[c];
        soa_arrays.hcn_channel_states[global_idx] = neuron.channels.hcn_h[c];
        
        // HH states
        soa_arrays.hh_m_states[global_idx] = (c == 0) ? neuron.m : neuron.m_comp[c];
        soa_arrays.hh_h_states[global_idx] = (c == 0) ? neuron.h : neuron.h_comp[c];
        soa_arrays.hh_n_states[global_idx] = (c == 0) ? neuron.n : neuron.n_comp[c];
    }
}

#endif // PERFORMANCE_OPTIMIZATION_H