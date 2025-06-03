#ifndef CUDA_COMPATIBILITY_H
#define CUDA_COMPATIBILITY_H

// CUDA Compatibility Header for NeuroGen Alpha
// Addresses C++14/17 type trait issues in CUDA compilation

#ifdef __CUDA_ARCH__
    // We're compiling device code
    #define CUDA_DEVICE_CODE
#endif

#ifdef __NVCC__
    // CUDA compiler specific fixes
    
    // Disable problematic C++17 features for CUDA compilation
    #if __cplusplus > 201402L
        #pragma message "Note: Using C++14 compatibility mode for CUDA"
    #endif
    
    // Provide missing type traits for CUDA context
    #ifndef __has_builtin
        #define __has_builtin(x) 0
    #endif
    
    // CUDA type trait compatibility
    namespace cuda_compat {
        template<typename T>
        struct is_array {
            static constexpr bool value = false;
        };
        
        template<typename T>
        struct is_array<T[]> {
            static constexpr bool value = true;
        };
        
        template<typename T, size_t N>
        struct is_array<T[N]> {
            static constexpr bool value = true;
        };
        
        template<typename T>
        struct is_reference {
            static constexpr bool value = false;
        };
        
        template<typename T>
        struct is_reference<T&> {
            static constexpr bool value = true;
        };
        
        template<typename T>
        struct is_reference<T&&> {
            static constexpr bool value = true;
        };
        
        template<typename T>
        struct is_object {
            static constexpr bool value = !is_reference<T>::value;
        };
        
        template<typename T, typename U>
        struct is_member_object_pointer {
            static constexpr bool value = false;
        };
        
        template<typename T, typename U>
        struct is_member_function_pointer {
            static constexpr bool value = false;
        };
        
        template<typename T>
        struct is_member_pointer {
            static constexpr bool value = false;
        };
    }
    
    // Override std type traits in CUDA context
    #ifdef CUDA_DEVICE_CODE
        namespace std {
            using cuda_compat::is_array;
            using cuda_compat::is_reference;
            using cuda_compat::is_object;
            using cuda_compat::is_member_object_pointer;
            using cuda_compat::is_member_function_pointer;
            using cuda_compat::is_member_pointer;
        }
    #endif
    
#endif // __NVCC__

// CUDA kernel launch configuration helpers
#ifdef __NVCC__
    
    // Safe block and grid size calculation
    inline __host__ dim3 makeSafeBlock(int size = 256) {
        return dim3(size, 1, 1);
    }
    
    inline __host__ dim3 makeSafeGrid(int total_threads, int block_size = 256) {
        int grid_size = (total_threads + block_size - 1) / block_size;
        return dim3(grid_size, 1, 1);
    }
    
    // Error checking macro
    #define CUDA_CHECK_ERROR(call) do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)
    
    // Synchronization helpers
    #define CUDA_SYNC_CHECK() CUDA_CHECK_ERROR(cudaDeviceSynchronize())
    
#else
    // Non-CUDA fallback definitions
    #define __host__
    #define __device__
    #define __global__
    #define __shared__
    #define CUDA_CHECK_ERROR(call) call
    #define CUDA_SYNC_CHECK() 
#endif

// Mathematical constants for both host and device
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef M_E
    #define M_E 2.71828182845904523536
#endif

// Device/Host function annotations
#ifdef __NVCC__
    #define CUDA_CALLABLE __host__ __device__
    #define CUDA_DEVICE __device__
    #define CUDA_HOST __host__
    #define CUDA_GLOBAL __global__
#else
    #define CUDA_CALLABLE
    #define CUDA_DEVICE
    #define CUDA_HOST
    #define CUDA_GLOBAL
#endif

// Memory management helpers
#ifdef __NVCC__
    template<typename T>
    T* cuda_malloc(size_t count) {
        T* ptr;
        CUDA_CHECK_ERROR(cudaMalloc(&ptr, count * sizeof(T)));
        return ptr;
    }
    
    template<typename T>
    void cuda_free(T* ptr) {
        if (ptr) {
            CUDA_CHECK_ERROR(cudaFree(ptr));
        }
    }
    
    template<typename T>
    void cuda_memcpy_h2d(T* dst, const T* src, size_t count) {
        CUDA_CHECK_ERROR(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void cuda_memcpy_d2h(T* dst, const T* src, size_t count) {
        CUDA_CHECK_ERROR(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
#endif

#endif // CUDA_COMPATIBILITY_H
