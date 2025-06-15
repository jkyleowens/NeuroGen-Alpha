// include/NeuroGen/MarketData.h
// Market data structure for cryptocurrency trading

#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <ostream>

/**
 * @brief Structure to hold market data for a cryptocurrency pair
 */
struct MarketData {
    std::string symbol;                               // Trading pair (e.g., "BTCUSD")
    float price;                                      // Current price
    float volume;                                     // Trading volume
    float open;                                       // Opening price
    float high;                                       // Highest price
    float low;                                        // Lowest price
    float close;                                      // Closing price
    std::chrono::system_clock::time_point timestamp;  // Data timestamp
    
    // Additional market indicators
    float bid;                                        // Bid price
    float ask;                                        // Ask price
    float spread;                                     // Bid-ask spread
    float volume_weighted_price;                      // VWAP
    float price_change_24h;                          // 24h price change
    float volume_change_24h;                         // 24h volume change
    
    // Constructor
    MarketData() 
        : price(0.0f)
        , volume(0.0f)
        , open(0.0f)
        , high(0.0f)
        , low(0.0f)
        , close(0.0f)
        , bid(0.0f)
        , ask(0.0f)
        , spread(0.0f)
        , volume_weighted_price(0.0f)
        , price_change_24h(0.0f)
        , volume_change_24h(0.0f)
        , timestamp(std::chrono::system_clock::now())
    {
    }
    
    // Constructor with basic data
    MarketData(const std::string& sym, float p, float v)
        : symbol(sym)
        , price(p)
        , volume(v)
        , open(p)
        , high(p)
        , low(p)
        , close(p)
        , bid(p * 0.999f)  // Approximate bid
        , ask(p * 1.001f)  // Approximate ask
        , spread(ask - bid)
        , volume_weighted_price(p)
        , price_change_24h(0.0f)
        , volume_change_24h(0.0f)
        , timestamp(std::chrono::system_clock::now())
    {
    }
    
    // Copy constructor
    MarketData(const MarketData& other) = default;
    
    // Assignment operator
    MarketData& operator=(const MarketData& other) = default;
    
    // Utility methods
    bool isValid() const {
        return price > 0.0f && volume >= 0.0f && !symbol.empty();
    }
    
    float getMidPrice() const {
        return (bid + ask) / 2.0f;
    }
    
    float getSpreadPercent() const {
        return (price > 0.0f) ? (spread / price) * 100.0f : 0.0f;
    }
    
    // Calculate volatility from OHLC
    float getIntraDayVolatility() const {
        if (low >= high || close <= 0.0f) return 0.0f;
        return (high - low) / close;
    }
    
    // Time since data was collected
    std::chrono::milliseconds getAge() const {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp);
    }
    
    // Check if data is stale (older than threshold)
    bool isStale(int max_age_seconds = 60) const {
        return getAge().count() > (max_age_seconds * 1000);
    }
    
    // Convert timestamp to string
    std::string getTimestampString() const {
        auto time_t = std::chrono::system_clock::to_time_t(timestamp);
        auto tm = *std::localtime(&time_t);
        char buffer[100];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
        return std::string(buffer);
    }
    
    // Print formatted market data
    void print() const {
        std::cout << "[MarketData] " << symbol 
                  << " - Price: $" << std::fixed << std::setprecision(2) << price
                  << ", Volume: " << volume
                  << ", Spread: " << std::setprecision(4) << getSpreadPercent() << "%"
                  << ", Time: " << getTimestampString() << std::endl;
    }
    
    // Operator for easy printing
    friend std::ostream& operator<<(std::ostream& os, const MarketData& data) {
        os << data.symbol << " $" << std::fixed << std::setprecision(2) << data.price 
           << " (Vol: " << data.volume << ")";
        return os;
    }
    
    // Comparison operators for sorting/searching
    bool operator<(const MarketData& other) const {
        return timestamp < other.timestamp;
    }
    
    bool operator>(const MarketData& other) const {
        return timestamp > other.timestamp;
    }
    
    bool operator==(const MarketData& other) const {
        return symbol == other.symbol && 
               std::abs(price - other.price) < 0.001f &&
               std::abs(volume - other.volume) < 0.001f;
    }
};

/**
 * @brief Collection of market data points for time series analysis
 */
class MarketDataSeries {
public:
    MarketDataSeries(const std::string& symbol, size_t max_size = 1000)
        : symbol_(symbol), max_size_(max_size) {
        data_.reserve(max_size_);
    }
    
    // Add new data point
    void addData(const MarketData& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push_back(data);
        
        // Remove old data if exceeding max size
        if (data_.size() > max_size_) {
            data_.erase(data_.begin());
        }
    }
    
    // Get latest data
    MarketData getLatest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.empty() ? MarketData() : data_.back();
    }
    
    // Get data range
    std::vector<MarketData> getRange(size_t count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) return {};
        
        size_t start = (data_.size() > count) ? data_.size() - count : 0;
        return std::vector<MarketData>(data_.begin() + start, data_.end());
    }
    
    // Get all data
    std::vector<MarketData> getAllData() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_;
    }
    
    // Calculate simple moving average
    float getMovingAverage(size_t periods) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.size() < periods) return 0.0f;
        
        float sum = 0.0f;
        for (size_t i = data_.size() - periods; i < data_.size(); i++) {
            sum += data_[i].price;
        }
        return sum / periods;
    }
    
    // Calculate price volatility
    float getVolatility(size_t periods) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.size() < periods + 1) return 0.0f;
        
        std::vector<float> returns;
        returns.reserve(periods);
        
        for (size_t i = data_.size() - periods; i < data_.size(); i++) {
            if (i > 0 && data_[i-1].price > 0) {
                float return_pct = (data_[i].price - data_[i-1].price) / data_[i-1].price;
                returns.push_back(return_pct);
            }
        }
        
        if (returns.empty()) return 0.0f;
        
        // Calculate standard deviation
        float mean = 0.0f;
        for (float r : returns) mean += r;
        mean /= returns.size();
        
        float variance = 0.0f;
        for (float r : returns) {
            float diff = r - mean;
            variance += diff * diff;
        }
        
        return std::sqrt(variance / returns.size());
    }
    
    // Get current size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }
    
    // Check if empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.empty();
    }
    
    // Clear all data
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }
    
private:
    std::string symbol_;
    size_t max_size_;
    std::vector<MarketData> data_;
    mutable std::mutex mutex_;
};

// Utility functions for market data analysis
namespace MarketDataUtils {
    
    // Calculate RSI (Relative Strength Index)
    float calculateRSI(const std::vector<MarketData>& data, size_t periods = 14);
    
    // Calculate MACD (Moving Average Convergence Divergence)
    std::pair<float, float> calculateMACD(const std::vector<MarketData>& data, 
                                         size_t fast_period = 12, 
                                         size_t slow_period = 26, 
                                         size_t signal_period = 9);
    
    // Calculate Bollinger Bands
    struct BollingerBands {
        float upper;
        float middle;
        float lower;
    };
    
    BollingerBands calculateBollingerBands(const std::vector<MarketData>& data, 
                                          size_t periods = 20, 
                                          float std_dev_multiplier = 2.0f);
    
    // Validate market data
    bool validateMarketData(const MarketData& data);
    
    // Convert between different time formats
    std::time_t timestampToTimeT(const std::chrono::system_clock::time_point& timestamp);
    std::chrono::system_clock::time_point timeTToTimestamp(std::time_t time_t);
    
    // Format price for display
    std::string formatPrice(float price, int decimals = 2);
    
    // Format volume for display
    std::string formatVolume(float volume);
}