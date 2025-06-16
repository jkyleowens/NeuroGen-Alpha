#ifndef NEUROGEN_KRAKENAPI_H
#define NEUROGEN_KRAKENAPI_H

#include <string>
#include <vector>
#include <chrono> // For std::chrono::system_clock::time_point potentially in MarketData

// Assuming PriceTick and MarketData structures are defined in these headers
// Adjust the paths if they are located elsewhere.
#include "NeuroGen/PriceTick.h"
#include "NeuroGen/MarketData.h"

// --- Kraken API Library Integration ---
// This header now relies on KrakenDataFetcher.h for the actual API communication.
// #include <krakenapicpp/krakenapi.h> // Example, replaced by KrakenDataFetcher
#include "NeuroGen/KrakenDataFetcher.h" // Include the actual fetcher header
// --- End Kraken API Library Integration ---

// Forward declaration for the API client object if not fully defined by KrakenDataFetcher.h
// For example, if KrakenDataFetcher is the main class, we don't need a separate krakenapi::KrakenAPI forward declaration.
// namespace krakenapi {
//     class KrakenAPI; 
// }

class KrakenApi {
public:
    KrakenApi();
    ~KrakenApi();

    /**
     * @brief Initializes the Kraken API client.
     * Uses KrakenDataFetcher for underlying operations.
     * @param api_key Your Kraken API key (may not be used if KrakenDataFetcher handles auth via its config).
     * @param api_secret Your Kraken API secret (similarly, may not be used directly here).
     * @param config_file_path Path to a configuration file (e.g., JSON) that might contain API keys
     *                         and other settings for KrakenDataFetcher.
     * @return True if initialization was successful, false otherwise.
     */
    bool initialize(const std::string& api_key = "", const std::string& api_secret = "", const std::string& config_file_path = "kraken_api_config.json");

    /**
     * @brief Fetches historical OHLC (Open, High, Low, Close) data for a given trading pair.
     * @param pair The asset pair to get OHLC data for (e.g., "XBTUSD", "ETHUSD").
     * @param since Optional. Return committed OHLC data since given Unix timestamp (exclusive). Defaults to 0 (all available data up to Kraken's limit).
     * @param interval Optional. Time frame interval in minutes. Defaults to "1".
     *                 Common values: "1", "5", "15", "30", "60" (1 hour), "240" (4 hours), "1440" (1 day), "10080" (1 week), "21600" (2 weeks).
     * @return A vector of PriceTick objects containing the historical data.
     */
    std::vector<PriceTick> fetchHistoricalData(
        const std::string& pair,
        long since = 0, // Unix timestamp (seconds)
        const std::string& interval = "1" // Interval in minutes as a string (e.g., "1", "60", "1440")
    );

    /**
     * @brief Converts a MarketData object (potentially from a live feed or different source) to a PriceTick object.
     * @param data The MarketData object to convert.
     * @return The corresponding PriceTick object.
     */
    PriceTick convertMarketDataToPriceTick(const MarketData& data);

    /**
     * @brief Converts a PriceTick object back to a MarketData object.
     * @param tick The PriceTick object to convert.
     * @param symbol The trading symbol for the MarketData object (e.g., "BTC/USD").
     * @return The corresponding MarketData object.
     */
    MarketData convertPriceTickToMarketData(const PriceTick& tick, const std::string& symbol);

private:
    // void* api_ptr_; // Now a KrakenDataFetcher*
    KrakenDataFetcher* api_fetcher_; // Pointer to the KrakenDataFetcher instance

    std::string api_key_;    // Store if needed for direct use or configuration
    std::string api_secret_; // Store if needed
    // std::string api_url_base_; // This would be part of KrakenDataFetcher's config

    // The createSignature method is likely no longer needed if KrakenDataFetcher handles auth.
    // std::string createSignature(
    //     const std::string& path,
    //     const std::string& nonce,
    //     const std::string& postdata
    // );
};

#endif // NEUROGEN_KRAKENAPI_H
