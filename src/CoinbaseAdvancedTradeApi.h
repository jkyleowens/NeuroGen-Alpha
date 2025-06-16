#ifndef NEUROGEN_COINBASEADVANCEDTRADEAPI_H
#define NEUROGEN_COINBASEADVANCEDTRADEAPI_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept> // For std::runtime_error
#include <chrono>    // For std::chrono::system_clock

#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include "NeuroGen/PriceTick.h" // Assuming PriceTick is in this path relative to include dirs

// Forward declaration if PriceTick is in a namespace or for complex dependencies
// namespace NeuroGen { struct PriceTick; }

class CoinbaseAdvancedTradeApi {
public:
    // Enum for candle granularities supported by Coinbase Advanced Trade API
    // Values correspond to strings like "ONE_MINUTE", "FIVE_MINUTE", etc.
    enum class CandleGranularity {
        UNKNOWN_GRANULARITY,
        ONE_MINUTE,
        FIVE_MINUTE,
        FIFTEEN_MINUTE,
        THIRTY_MINUTE,
        ONE_HOUR,
        TWO_HOUR,
        SIX_HOUR,
        ONE_DAY
    };

    CoinbaseAdvancedTradeApi();
    ~CoinbaseAdvancedTradeApi();

    /**
     * @brief Initializes the API client with credentials and base URL.
     * @param api_key Your Coinbase API Key.
     * @param api_secret Your Coinbase API Secret.
     * @param base_url The base URL for the Coinbase Advanced Trade API (e.g., "https://api.coinbase.com").
     * @return True if initialization is successful, false otherwise.
     */
    bool initialize(const std::string& api_key, const std::string& api_secret, const std::string& base_url = "https://api.coinbase.com");

    /**
     * @brief Fetches historical candle data for a specific product.
     * @param product_id The trading pair (e.g., "BTC-USD").
     * @param start_timestamp_s Unix timestamp in seconds for the start of the period.
     * @param end_timestamp_s Unix timestamp in seconds for the end of the period.
     * @param granularity The granularity of the candles.
     * @return A vector of PriceTick objects. Throws std::runtime_error on failure.
     */
    std::vector<PriceTick> get_product_candles(
        const std::string& product_id,
        long long start_timestamp_s,
        long long end_timestamp_s,
        CandleGranularity granularity);

    /**
     * @brief Fetches the current server time from Coinbase.
     * Useful for synchronizing requests and ensuring timestamp accuracy for signatures.
     * @return Server time as a Unix timestamp in seconds. Returns -1 on failure.
     */
    long long get_server_time();

    /**
     * @brief Tests connectivity to the API and validity of credentials (e.g., by fetching server time or a simple endpoint).
     * @return True if connection and authentication (if tested) are successful, false otherwise.
     */
    bool test_connectivity();

private:
    std::string api_key_;
    std::string api_secret_;
    std::string base_url_;
    CURL* curl_handle_;
    bool initialized_;

    /**
     * @brief Generates the HMAC-SHA256 signature for API requests.
     * Coinbase signature: HMAC-SHA256(timestamp + method + requestPath + body, api_secret)
     * @param timestamp Current Unix timestamp as a string.
     * @param method HTTP method (e.g., "GET", "POST").
     * @param request_path The API endpoint path (e.g., "/api/v3/brokerage/products").
     * @param body The request body (empty string for GET requests).
     * @return The hex-encoded signature string.
     */
    std::string _generate_signature(const std::string& timestamp, const std::string& method, const std::string& request_path, const std::string& body);

    /**
     * @brief Makes an HTTP request to the Coinbase API.
     * @param method HTTP method (e.g., "GET", "POST").
     * @param endpoint The API endpoint path (e.g., "/api/v3/brokerage/products/{product_id}/candles").
     * @param params Query parameters for GET requests or JSON body for POST requests.
     * @param needs_auth Whether the request requires authentication headers.
     * @param response_string String to store the response body.
     * @return HTTP status code.
     */
    long _make_request(const std::string& method, const std::string& endpoint, const std::string& params_or_body, bool needs_auth, std::string& response_string);

    /**
     * @brief Converts CandleGranularity enum to its string representation for the API.
     */
    std::string _granularity_to_string(CandleGranularity granularity);

    /**
     * @brief Parses the JSON response from get_product_candles into PriceTick objects.
     */
    std::vector<PriceTick> _parse_candles_response(const nlohmann::json& json_response);

    /**
     * @brief libcurl write callback function.
     */
    static size_t _curl_write_callback(void* contents, size_t size, size_t nmemb, std::string* userp);
};

#endif // NEUROGEN_COINBASEADVANCEDTRADEAPI_H
