#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// Always include real CUDA headers when building with nvcc
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

// Include compatibility header for additional utilities
#include "CudaCompatibility.h"
// Include existing grid utilities to avoid redefinition
#include "GridBlockUtils.cuh"

// Forward declarations for GPU structures
struct GPUNeuronState;
struct GPUSynapse;
struct GPUSpikeEvent;

// CUDA Grid and Block Utilities
namespace CudaUtils {
    
    // Standard block size for CUDA kernels
    const int DEFAULT_BLOCK_SIZE = 256;
    const int MAX_BLOCKS = 65535;
    
    /**
     * Creates a safe CUDA block dimension with bounds checking
     * @param size Block size (default: 256)
     * @return dim3 block dimensions
     */
    inline __host__ dim3 makeSafeBlock(int size = DEFAULT_BLOCK_SIZE) {
        // Ensure block size is within valid range
        if (size <= 0) size = DEFAULT_BLOCK_SIZE;
        if (size > 1024) size = 1024; // Maximum threads per block
        return dim3(size);
    }
    
    /**
     * Creates a safe CUDA grid dimension with bounds checking
     * @param total_threads Total number of threads needed
     * @param block_size Block size (default: 256)
     * @return dim3 grid dimensions
     */
    inline __host__ dim3 makeSafeGrid(int total_threads, int block_size = DEFAULT_BLOCK_SIZE) {
        if (total_threads <= 0) return dim3(1);
        if (block_size <= 0) block_size = DEFAULT_BLOCK_SIZE;
        
        int grid_size = (total_threads + block_size - 1) / block_size;
        // Ensure we don't exceed maximum grid size
        if (grid_size > MAX_BLOCKS) {
            grid_size = MAX_BLOCKS;
        }
        return dim3(grid_size);
    }
    
    /**
     * Get optimal block size for a given number of threads
     * @param num_threads Number of threads
     * @return Optimal block size
     */
    inline __host__ int getOptimalBlockSize(int num_threads) {
        if (num_threads >= 512) return 512;
        if (num_threads >= 256) return 256;
        if (num_threads >= 128) return 128;
        if (num_threads >= 64) return 64;
        return 32;
    }
    
    /**
     * Calculate shared memory size needed
     * @param threads_per_block Threads per block
     * @param bytes_per_thread Bytes per thread
     * @return Total shared memory size in bytes
     */
    inline __host__ size_t calculateSharedMemorySize(int threads_per_block, int bytes_per_thread) {
        return threads_per_block * bytes_per_thread;
    }
}

// Global convenience functions (for backward compatibility with safe versions only)
// Note: makeBlock() and makeGrid() are provided by GridBlockUtils.cuh
inline __host__ dim3 makeSafeBlock(int size = CudaUtils::DEFAULT_BLOCK_SIZE) {
    return CudaUtils::makeSafeBlock(size);
}

inline __host__ dim3 makeSafeGrid(int total_threads, int block_size = CudaUtils::DEFAULT_BLOCK_SIZE) {
    return CudaUtils::makeSafeGrid(total_threads, block_size);
}

// CUDA Error Checking Utilities
namespace CudaErrorUtils {
    
    /**
     * Check CUDA error and print message if error occurred
     * @param err CUDA error code
     * @param file Source file name
     * @param line Line number
     * @return true if no error, false if error occurred
     */
    inline bool checkError(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
    
    /**
     * Check last CUDA error from kernel launch
     * @param file Source file name
     * @param line Line number
     * @return true if no error, false if error occurred
     */
    inline bool checkKernelError(const char* file, int line) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
    
    /**
     * Synchronize device and check for errors
     * @param file Source file name
     * @param line Line number
     * @return true if no error, false if error occurred
     */
    inline bool syncAndCheck(const char* file, int line) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
}

// Enhanced error checking macros
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (!CudaErrorUtils::checkError(err, __FILE__, __LINE__)) { \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL_ERROR() \
    do { \
        if (!CudaErrorUtils::checkKernelError(__FILE__, __LINE__)) { \
            exit(1); \
        } \
    } while(0)

#define CUDA_SYNC_AND_CHECK() \
    do { \
        if (!CudaErrorUtils::syncAndCheck(__FILE__, __LINE__)) { \
            exit(1); \
        } \
    } while(0)

// Memory Management Utilities
namespace CudaMemoryUtils {
    
    /**
     * Allocate device memory with error checking
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory or nullptr on failure
     */
    template<typename T>
    inline T* allocateDevice(size_t count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate %zu bytes on device: %s\n", 
                    count * sizeof(T), cudaGetErrorString(err));
            return nullptr;
        }
        return ptr;
    }
    
    /**
     * Free device memory with error checking
     * @param ptr Pointer to memory to free
     */
    template<typename T>
    inline void freeDevice(T* ptr) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to free device memory: %s\n", cudaGetErrorString(err));
            }
        }
    }
    
    /**
     * Copy data from host to device
     * @param dst Destination device pointer
     * @param src Source host pointer
     * @param count Number of elements to copy
     * @return true on success, false on failure
     */
    template<typename T>
    inline bool copyHostToDevice(T* dst, const T* src, size_t count) {
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
        return CudaErrorUtils::checkError(err, __FILE__, __LINE__);
    }
    
    /**
     * Copy data from device to host
     * @param dst Destination host pointer
     * @param src Source device pointer
     * @param count Number of elements to copy
     * @return true on success, false on failure
     */
    template<typename T>
    inline bool copyDeviceToHost(T* dst, const T* src, size_t count) {
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
        return CudaErrorUtils::checkError(err, __FILE__, __LINE__);
    }
}

// Device Information Utilities
namespace CudaDeviceUtils {
    
    /**
     * Get CUDA device properties
     * @param device Device ID (default: 0)
     * @return Device properties structure
     */
    inline cudaDeviceProp getDeviceProperties(int device = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return prop;
    }
    
    /**
     * Get number of CUDA devices
     * @return Number of available CUDA devices
     */
    inline int getDeviceCount() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
    
    /**
     * Set the current CUDA device
     * @param device Device ID to set
     * @return true on success, false on failure
     */
    inline bool setDevice(int device) {
        cudaError_t err = cudaSetDevice(device);
        return CudaErrorUtils::checkError(err, __FILE__, __LINE__);
    }
    
    /**
     * Print device information
     * @param device Device ID (default: 0)
     */
    inline void printDeviceInfo(int device = 0) {
        cudaDeviceProp prop = getDeviceProperties(device);
        printf("Device %d: %s\n", device, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp size: %d\n", prop.warpSize);
    }
}

#endif // CUDA_UTILS_H
