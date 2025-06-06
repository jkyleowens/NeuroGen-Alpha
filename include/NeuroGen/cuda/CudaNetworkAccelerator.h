#ifndef CUDA_NETWORK_ACCELERATOR_H
#define CUDA_NETWORK_ACCELERATOR_H

/**
 * @brief Minimal stub for the CUDA network accelerator.
 *
 * This placeholder is provided so that CPU builds can compile
 * even when the full CUDA implementation is unavailable.
 */
class CudaNetworkAccelerator {
public:
    CudaNetworkAccelerator() = default;
    ~CudaNetworkAccelerator() = default;
};

#endif // CUDA_NETWORK_ACCELERATOR_H
