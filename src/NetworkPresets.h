#ifndef NEUROGENALPHA_NETWORKPRESETS_H
#define NEUROGENALPHA_NETWORKPRESETS_H

#include "NetworkConfig.h"

/**
 * @brief NetworkPresets namespace provides predefined network configurations
 * for common use cases like trading, research, and testing scenarios.
 */
namespace NetworkPresets {

    /**
     * @brief Trading-optimized network configuration
     * Designed for financial trading applications with balanced performance
     * - Input size: 64 (market indicators, price data, volume, etc.)
     * - Hidden size: 256 (sufficient capacity for pattern recognition)
     * - Output size: 10 (trading decisions for multiple assets)
     * - Optimized learning rates and connectivity for financial data
     * @return NetworkConfig optimized for trading tasks
     */
    NetworkConfig trading_optimized();

    /**
     * @brief High-frequency trading configuration
     * Optimized for low-latency, high-speed trading decisions
     * - Input size: 32 (essential market data only)
     * - Hidden size: 128 (smaller for speed)
     * - Output size: 5 (simple buy/sell decisions)
     * - Ultra-fast timing and aggressive learning parameters
     * @return NetworkConfig optimized for HFT applications
     */
    NetworkConfig high_frequency_trading();

    /**
     * @brief Research-detailed configuration
     * Large network for detailed research and experimentation
     * - Input size: 128 (comprehensive input space)
     * - Hidden size: 512 (large hidden layer for complex patterns)
     * - Output size: 20 (detailed output classifications)
     * - Conservative learning for detailed study
     * @return NetworkConfig with extensive parameters for research
     */
    NetworkConfig research_detailed();

    /**
     * @brief Minimal test configuration
     * Small network for quick testing and debugging
     * - Input size: 8 (minimal input)
     * - Hidden size: 16 (small hidden layer)
     * - Output size: 3 (simple output)
     * - Fast testing timing with standard parameters
     * @return NetworkConfig with minimal parameters for testing
     */
    NetworkConfig minimal_test();

    /**
     * @brief Small network configuration for testing
     * Compact network suitable for testing and development
     * - Input size: 16 (small input)
     * - Hidden size: 32 (small hidden layer)
     * - Output size: 4 (limited output)
     * - Standard timing parameters for quick testing
     * @return NetworkConfig with small parameters for testing
     */
    NetworkConfig getSmallNetworkConfig();

    /**
     * @brief Balanced default configuration
     * General-purpose configuration suitable for most applications
     * - Input size: 32 (moderate input size)
     * - Hidden size: 128 (balanced hidden layer)
     * - Output size: 8 (reasonable output size)
     * - Standard timing and balanced learning parameters
     * @return NetworkConfig with balanced parameters
     */
    NetworkConfig balanced_default();

} // namespace NetworkPresets

#endif // NEUROGENALPHA_NETWORKPRESETS_H
