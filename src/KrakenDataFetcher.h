// include/NeuroGen/KrakenDataFetcher.h
// Kraken API data fetcher for cryptocurrency trading

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <atomic>
#include <map>
#include <functional>
#include "MarketData.h"

// Forward declarations
struct CURL;
typedef struct CURL CURL;

/**
 * @brief HTTP response structure
 */
struct HttpResponse {
    std::string data;
    long status_code;
    std::string error_message;
    std::chrono::milliseconds response_time;
    
    HttpResponse() : status_code(0), response_time(0) {}
    
    bool isSuccess() const {
        return status_code >= 200 && status_code < 300 && error_message.empty();
    }
};

/**
 * @brief OHLC (Open, High, Low, Close) data structure
 */
struct OHLCData {
    std::chrono::system_clock::time_point timestamp;
    float open;
    float high;
    float low;
    float close;
    float volume;
    float vwap;  // Volume Weighted Average Price
    int trades;  // Number of trades
    
    OHLCData() : open(0), high(0), low(0), close(0), volume(0), vwap(0), trades(0) {}
    
    bool isValid() const {
        return open > 0 && high > 0 && low > 0 && close > 0 && 
               high >= low && high >= open && high >= close && 
               low <= open && low <= close;
    }
    
    MarketData toMarketData(const std::string& symbol) const {
        MarketData data;
        data.symbol = symbol;
        data.price = close;
        data.volume = volume;
        data.open = open;
        data.high = high;
        data.low = low;
        data.close = close;
        data.volume_weighted_price = vwap;
        data.timestamp = timestamp;
        data.bid = close * 0.999f;  // Approximate
        data.ask = close * 1.001f;  // Approximate
        data.spread = data.ask - data.bid;
        return data;
    }
};

/**
 * @brief Ticker information structure
 */
struct TickerData {
    std::string symbol;
    float last_price;
    float bid;
    float ask;
    float volume_24h;
    float volume_weighted_avg_24h;
    float high_24h;
    float low_24h;
    float open_24h;
    int trades_24h;
    std::chrono::system_clock::time_point timestamp;
    
    TickerData() : last_price(0), bid(0), ask(0), volume_24h(0), 
                   volume_weighted_avg_24h(0), high_24h(0), low_24h(0), 
                   open_24h(0), trades_24h(0) {}
    
    MarketData toMarketData() const {
        MarketData data;
        data.symbol = symbol;
        data.price = last_price;
        data.volume = volume_24h;
        data.open = open_24h;
        data.high = high_24h;
        data.low = low_24h;
        data.close = last_price;
        data.bid = bid;
        data.ask = ask;
        data.spread = ask - bid;
        data.volume_weighted_price = volume_weighted_avg_24h;
        data.price_change_24h = (last_price - open_24h) / open_24h;
        data.timestamp = timestamp;
        return data;
    }
};

/**
 * @brief Rate limiting configuration
 */
struct RateLimitConfig {
    int max_requests_per_minute;
    int max_requests_per_second;
    std::chrono::milliseconds min_request_interval;
    bool strict_mode;
    
    RateLimitConfig() 
        : max_requests_per_minute(60)
        , max_requests_per_second(2)
        , min_request_interval(500)  // 500ms minimum between requests
        , strict_mode(true) {}
};

/**
 * @brief Kraken API configuration
 */
struct KrakenConfig {
    std::string base_url;
    std::string api_version;
    std::string user_agent;
    int timeout_seconds;
    int max_retries;
    RateLimitConfig rate_limit;
    bool use_ssl_verification;
    
    KrakenConfig() 
        : base_url("https://api.kraken.com")
        , api_version("0")
        , user_agent("NeuroGen-Alpha-Crypto-Trader/2.1")
        , timeout_seconds(10)
        , max_retries(3)
        , use_ssl_verification(true) {}
};

/**
 * @brief Main Kraken API data fetcher class
 */
class KrakenDataFetcher {
public:
    // Constructor and destructor
    explicit KrakenDataFetcher(const KrakenConfig& config = KrakenConfig());
    ~KrakenDataFetcher();
    
    // Initialization and cleanup
    bool initialize();
    void cleanup();
    bool isInitialized() const { return is_initialized_; }
    
    // Market data fetching methods
    bool fetchCurrentPrice(const std::string& symbol, float& price);
    bool fetchTickerData(const std::string& symbol, TickerData& ticker);
    bool fetchMarketData(const std::string& symbol, MarketData& data);
    bool fetchOHLCData(const std::string& symbol, std::vector<OHLCData>& ohlc_data, 
                       int interval_minutes = 1, int count = 50);
    
    // Batch operations
    bool fetchMultipleSymbols(const std::vector<std::string>& symbols, 
                             std::map<std::string, MarketData>& results);
    
    // Historical data
    bool fetchHistoricalData(const std::string& symbol, 
                            std::chrono::system_clock::time_point start_time,
                            std::chrono::system_clock::time_point end_time,
                            std::vector<OHLCData>& historical_data,
                            int interval_minutes = 1);
    
    // Real-time streaming (polling-based)
    bool startRealTimeData(const std::string& symbol, 
                          std::function<void(const MarketData&)> callback,
                          int update_interval_seconds = 60);
    void stopRealTimeData();
    bool isRealTimeActive() const { return is_real_time_active_; }
    
    // Symbol management
    std::vector<std::string> getSupportedSymbols();
    bool isSymbolSupported(const std::string& symbol);
    std::string normalizeSymbol(const std::string& input_symbol);
    
    // Configuration and status
    void setConfig(const KrakenConfig& config);
    KrakenConfig getConfig() const { return config_; }
    void setRateLimit(const RateLimitConfig& rate_limit);
    
    // Error handling and diagnostics
    std::string getLastError() const { return last_error_; }
    int getLastHttpStatus() const { return last_http_status_; }
    std::chrono::milliseconds getLastResponseTime() const { return last_response_time_; }
    
    // Statistics
    struct Statistics {
        int total_requests;
        int successful_requests;
        int failed_requests;
        int rate_limited_requests;
        std::chrono::milliseconds total_response_time;
        std::chrono::milliseconds avg_response_time;
        std::chrono::system_clock::time_point last_request_time;
        
        Statistics() : total_requests(0), successful_requests(0), failed_requests(0),
                      rate_limited_requests(0), total_response_time(0), avg_response_time(0) {}
    };
    
    Statistics getStatistics() const { return stats_; }
    void resetStatistics();
    
    // Utility methods
    bool testConnection();
    std::string getServerTime();
    
private:
    // Configuration
    KrakenConfig config_;
    bool is_initialized_;
    
    // HTTP client
    CURL* curl_handle_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::vector<std::chrono::steady_clock::time_point> request_history_;
    std::mutex rate_limit_mutex_;
    
    // Real-time data
    std::atomic<bool> is_real_time_active_;
    std::thread real_time_thread_;
    std::function<void(const MarketData&)> real_time_callback_;
    std::string real_time_symbol_;
    int real_time_interval_;
    
    // Error tracking
    std::string last_error_;
    int last_http_status_;
    std::chrono::milliseconds last_response_time_;
    
    // Statistics
    Statistics stats_;
    mutable std::mutex stats_mutex_;
    
    // Symbol mappings (Kraken uses different symbol names)
    std::map<std::string, std::string> symbol_mapping_;
    
    // Internal methods
    void initializeSymbolMapping();
    void initializeCurl();
    void cleanupCurl();
    
    // HTTP operations
    HttpResponse makeHttpRequest(const std::string& endpoint, 
                                const std::map<std::string, std::string>& params = {});
    std::string buildUrl(const std::string& endpoint) const;
    std::string buildQueryString(const std::map<std::string, std::string>& params) const;
    
    // Rate limiting
    bool checkRateLimit();
    void updateRateLimit();
    void waitForRateLimit();
    
    // JSON parsing
    bool parseTickerResponse(const std::string& json_response, 
                            const std::string& symbol, 
                            TickerData& ticker);
    bool parseOHLCResponse(const std::string& json_response, 
                          std::vector<OHLCData>& ohlc_data);
    bool parseErrorResponse(const std::string& json_response, std::string& error_message);
    
    // Data validation
    bool validateTickerData(const TickerData& ticker) const;
    bool validateOHLCData(const OHLCData& ohlc) const;
    bool validateMarketData(const MarketData& data) const;
    
    // Real-time data thread
    void realTimeDataLoop();
    
    // Statistics update
    void updateStatistics(bool success, std::chrono::milliseconds response_time);
    
    // Error handling
    void setLastError(const std::string& error);
    void logError(const std::string& operation, const std::string& error) const;
    void logInfo(const std::string& message) const;
    
    // Callback for libcurl
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp);
};

/**
 * @brief Factory class for creating data fetchers
 */
class DataFetcherFactory {
public:
    static std::unique_ptr<KrakenDataFetcher> createKrakenFetcher(const KrakenConfig& config = KrakenConfig());
    static bool testKrakenConnection(const KrakenConfig& config = KrakenConfig());
};

/**
 * @brief Utility functions for Kraken API integration
 */
namespace KrakenUtils {
    
    // Symbol conversion utilities
    std::string convertToKrakenSymbol(const std::string& standard_symbol);
    std::string convertFromKrakenSymbol(const std::string& kraken_symbol);
    bool isValidSymbolFormat(const std::string& symbol);
    
    // Time utilities
    std::string timestampToKrakenTime(const std::chrono::system_clock::time_point& timestamp);
    std::chrono::system_clock::time_point krakenTimeToTimestamp(const std::string& kraken_time);
    
    // Data validation
    bool validatePriceData(float price);
    bool validateVolumeData(float volume);
    bool validateTimestampData(const std::chrono::system_clock::time_point& timestamp);
    
    // Error handling
    std::string getKrakenErrorDescription(const std::string& error_code);
    bool isRetryableError(const std::string& error_code);
    
    // Configuration helpers
    KrakenConfig getDefaultConfig();
    KrakenConfig getHighFrequencyConfig();
    KrakenConfig getConservativeConfig();
    
    // JSON utilities (if using a JSON library)
    bool extractJsonField(const std::string& json, const std::string& field, std::string& value);
    bool extractJsonFloat(const std::string& json, const std::string& field, float& value);
    bool extractJsonInt(const std::string& json, const std::string& field, int& value);
}