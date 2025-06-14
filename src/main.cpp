#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <thread>
#include <csignal> // For handling Ctrl+C
#include <cstdlib> // For std::system
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <curl/curl.h>
#include <regex>

// NeuroGen CUDA Interface
#include <NeuroGen/cuda/NetworkCUDA_Interface.h>
#include <NeuroGen/TradingAgent.h>

// Forward declarations
class KrakenAPIClient;
class CryptoTradingSimulator;
class RealTimeMonitor;

// =============================================================================
// KRAKEN API CLIENT FOR REAL-TIME CRYPTO DATA
// =============================================================================

// Callback function for libcurl to write response data
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *response) {
    size_t totalSize = size * nmemb;
    response->append((char*)contents, totalSize);
    return totalSize;
}

class KrakenAPIClient {
private:
    CURL* curl_;
    std::string base_url_;
    std::chrono::steady_clock::time_point last_request_time_;
    static constexpr double MIN_REQUEST_INTERVAL = 1.0; // 1 second between requests
    
    // Symbol mapping from generic to Kraken format
    std::map<std::string, std::string> symbol_map_ = {
        {"BTCUSD", "XXBTZUSD"},
        {"ETHUSD", "XETHZUSD"},
        {"ADAUSD", "ADAUSD"},
        {"SOLUSD", "SOLUSD"},
        {"DOTUSD", "DOTUSD"},
        {"LINKUSD", "LINKUSD"}
    };
    
public:
    KrakenAPIClient() : base_url_("https://api.kraken.com/0/public/") {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_ = curl_easy_init();
        if (!curl_) {
            throw std::runtime_error("Failed to initialize libcurl");
        }
        
        // Set common curl options
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl_, CURLOPT_USERAGENT, "NeuroGen-Alpha-Trading-Bot/1.0");
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        
        last_request_time_ = std::chrono::steady_clock::now();
        
        std::cout << "[KRAKEN] API client initialized" << std::endl;
    }
    
    ~KrakenAPIClient() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
        curl_global_cleanup();
    }
    
    void enforceRateLimit() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_request_time_).count();
        
        if (elapsed < MIN_REQUEST_INTERVAL) {
            double sleep_time = MIN_REQUEST_INTERVAL - elapsed;
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
        }
        
        last_request_time_ = std::chrono::steady_clock::now();
    }
    
    std::string makeRequest(const std::string& endpoint) {
        enforceRateLimit();
        
        std::string url = base_url_ + endpoint;
        std::string response;
        
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        CURLcode res = curl_easy_perform(curl_);
        
        if (res != CURLE_OK) {
            throw std::runtime_error("Curl request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        long response_code;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);
        
        if (response_code != 200) {
            throw std::runtime_error("HTTP error: " + std::to_string(response_code));
        }
        
        return response;
    }
    
    MarketData getCurrentPrice(const std::string& symbol) {
        try {
            std::string kraken_symbol = getKrakenSymbol(symbol);
            std::string endpoint = "Ticker?pair=" + kraken_symbol;
            
            std::cout << "[KRAKEN] Fetching current price for " << symbol << " (" << kraken_symbol << ")" << std::endl;
            
            std::string response = makeRequest(endpoint);
            
            // Parse JSON response manually (simple approach)
            MarketData data = parseTickerResponse(response, symbol);
            
            std::cout << "[KRAKEN] " << symbol << " Price: $" << std::fixed << std::setprecision(2) 
                      << data.close << " Volume: " << data.volume << std::endl;
            
            return data;
            
        } catch (const std::exception& e) {
            std::cerr << "[KRAKEN ERROR] " << e.what() << std::endl;
            std::cout << "[KRAKEN] Falling back to synthetic data..." << std::endl;
            return generateFallbackData(symbol);
        }
    }
    
    std::vector<MarketData> getOHLCData(const std::string& symbol, int interval = 60, int count = 50) {
        try {
            std::string kraken_symbol = getKrakenSymbol(symbol);
            std::string endpoint = "OHLC?pair=" + kraken_symbol + "&interval=" + std::to_string(interval);
            
            std::cout << "[KRAKEN] Fetching OHLC data for " << symbol << std::endl;
            
            std::string response = makeRequest(endpoint);
            
            return parseOHLCResponse(response, symbol, count);
            
        } catch (const std::exception& e) {
            std::cerr << "[KRAKEN ERROR] " << e.what() << std::endl;
            std::cout << "[KRAKEN] Generating synthetic historical data..." << std::endl;
            return generateSyntheticHistoricalData(symbol, count);
        }
    }
    
    // Public method for generating fallback data when API fails
    MarketData generateFallbackData(const std::string& symbol) {
        // Base prices for different cryptocurrencies
        static std::map<std::string, float> base_prices = {
            {"BTCUSD", 45000.0f},
            {"ETHUSD", 3000.0f},
            {"ADAUSD", 0.8f},
            {"SOLUSD", 150.0f},
            {"DOTUSD", 25.0f},
            {"LINKUSD", 15.0f}
        };
        
        float base_price = base_prices.count(symbol) ? base_prices[symbol] : 1000.0f;
        
        // Add some random variation
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> price_variation(1.0f, 0.02f); // 2% volatility
        
        float current_price = base_price * price_variation(gen);
        
        MarketData data;
        data.datetime = std::to_string(std::time(nullptr));
        data.open = current_price * 0.999f;
        data.high = current_price * 1.005f;
        data.low = current_price * 0.995f;
        data.close = current_price;
        data.volume = 1000000.0f * std::abs(price_variation(gen));
        
        return data;
    }
    
private:
    std::string getKrakenSymbol(const std::string& symbol) {
        auto it = symbol_map_.find(symbol);
        if (it != symbol_map_.end()) {
            return it->second;
        }
        return symbol; // Use as-is if not in mapping
    }
    
    MarketData parseTickerResponse(const std::string& response, const std::string& symbol) {
        (void)symbol; // Mark as unused to suppress warning
        MarketData data;
        data.datetime = std::to_string(std::time(nullptr));
        
        // Simple regex-based JSON parsing for ticker data
        // Looking for pattern: "c":["price","volume"]
        std::regex price_regex(R"("c":\[\"([0-9.]+)\",\"([0-9.]+)\"\])");
        std::regex volume_regex(R"("v":\[\"([0-9.]+)\",\"([0-9.]+)\"\])");
        std::regex high_regex(R"("h":\[\"([0-9.]+)\",\"([0-9.]+)\"\])");
        std::regex low_regex(R"("l":\[\"([0-9.]+)\",\"([0-9.]+)\"\])");
        std::regex open_regex(R"("o":\"([0-9.]+)\")");
        
        std::smatch match;
        
        // Extract current price (close)
        if (std::regex_search(response, match, price_regex)) {
            data.close = std::stof(match[1].str());
            data.open = data.close * 0.999f; // Approximate
        } else {
            throw std::runtime_error("Could not parse price from response");
        }
        
        // Extract volume
        if (std::regex_search(response, match, volume_regex)) {
            data.volume = std::stof(match[2].str()); // 24h volume
        } else {
            data.volume = 1000000.0f; // Default volume
        }
        
        // Extract high/low
        if (std::regex_search(response, match, high_regex)) {
            data.high = std::stof(match[2].str()); // 24h high
        } else {
            data.high = data.close * 1.02f;
        }
        
        if (std::regex_search(response, match, low_regex)) {
            data.low = std::stof(match[2].str()); // 24h low
        } else {
            data.low = data.close * 0.98f;
        }
        
        return data;
    }
    
    std::vector<MarketData> parseOHLCResponse(const std::string& response, const std::string& symbol, int count) {
        std::vector<MarketData> result;
        
        // Simple OHLC parsing - look for array patterns
        // Kraken OHLC format: [timestamp, open, high, low, close, vwap, volume, count]
        std::regex ohlc_regex(R"(\[([0-9]+),\"([0-9.]+)\",\"([0-9.]+)\",\"([0-9.]+)\",\"([0-9.]+)\",\"([0-9.]+)\",\"([0-9.]+)\",([0-9]+)\])");
        
        auto search_start = response.cbegin();
        std::smatch match;
        
        while (std::regex_search(search_start, response.cend(), match, ohlc_regex) && result.size() < static_cast<size_t>(count)) {
            MarketData data;
            data.datetime = match[1].str();
            data.open = std::stof(match[2].str());
            data.high = std::stof(match[3].str());
            data.low = std::stof(match[4].str());
            data.close = std::stof(match[5].str());
            data.volume = std::stof(match[7].str());
            
            if (data.validate()) {
                result.push_back(data);
            }
            
            search_start = match.suffix().first;
        }
        
        if (result.empty()) {
            std::cout << "[KRAKEN] No OHLC data parsed, generating synthetic data" << std::endl;
            return generateSyntheticHistoricalData(symbol, count);
        }
        
        std::cout << "[KRAKEN] Parsed " << result.size() << " OHLC data points" << std::endl;
        return result;
    }
    
    std::vector<MarketData> generateSyntheticHistoricalData(const std::string& symbol, int count) {
        std::vector<MarketData> result;
        
        float base_price = 45000.0f; // Default to Bitcoin price
        if (symbol == "ETHUSD") base_price = 3000.0f;
        else if (symbol == "ADAUSD") base_price = 0.8f;
        else if (symbol == "SOLUSD") base_price = 150.0f;
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> price_change(0.0f, 0.01f); // 1% volatility
        
        float current_price = base_price;
        auto current_time = std::time(nullptr);
        
        for (int i = count - 1; i >= 0; --i) {
            MarketData data;
            data.datetime = std::to_string(current_time - (i * 3600)); // 1 hour intervals
            
            // Apply price change
            current_price *= (1.0f + price_change(gen));
            
            data.open = current_price * 0.999f;
            data.high = current_price * 1.002f;
            data.low = current_price * 0.998f;
            data.close = current_price;
            data.volume = 1000000.0f * (1.0f + std::abs(price_change(gen)));
            
            result.push_back(data);
        }
        
        std::cout << "[KRAKEN] Generated " << result.size() << " synthetic historical data points" << std::endl;
        return result;
    }
};

// =============================================================================
// REAL-TIME MONITORING AND DASHBOARD
// =============================================================================

class RealTimeMonitor {
private:
    std::map<std::string, float> latest_prices_;
    std::map<std::string, float> latest_volumes_;
    std::chrono::system_clock::time_point start_time_;
    std::mutex data_mutex_;
    
public:
    RealTimeMonitor() : start_time_(std::chrono::system_clock::now()) {}
    
    void updateMarketData(const std::string& symbol, const MarketData& data) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_prices_[symbol] = data.close;
        latest_volumes_[symbol] = data.volume;
    }
    
    void printStatus(const TradingAgent& agent, const std::string& symbol) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "NeuroGen-Alpha Crypto Trading Monitor" << std::endl;
        std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (latest_prices_.count(symbol)) {
            std::cout << symbol << " Price: $" << std::fixed << std::setprecision(2) 
                      << latest_prices_[symbol] << std::endl;
            std::cout << symbol << " Volume: " << latest_volumes_[symbol] << std::endl;
        }
        
        agent.printStatus();
        std::cout << std::string(60, '=') << std::endl;
    }
};

// =============================================================================
// CHART GENERATOR FOR REAL-TIME VISUALIZATION
// =============================================================================

class ChartGenerator {
private:
    struct DataPoint {
        double time;
        double value;
    };
    std::map<std::string, std::vector<DataPoint>> data_series_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::mutex chart_mutex_;

public:
    ChartGenerator() : start_time_(std::chrono::steady_clock::now()) {}

    void record(const std::string& series_name, double value) {
        std::lock_guard<std::mutex> lock(chart_mutex_);
        double time_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
        data_series_[series_name].push_back({time_elapsed, value});
    }

    void generateDashboard(const std::string& filename) {
        std::lock_guard<std::mutex> lock(chart_mutex_);
        
        std::cout << "\n[CHARTS] Generating cryptocurrency trading dashboard: " << filename << std::endl;
        std::ofstream file(filename);
        
        file << R"(<!DOCTYPE html><html><head><title>NeuroGen-Alpha Crypto Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
                h1 { text-align: center; color: #fff; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 2.5em; margin-bottom: 30px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }
                .chart { background: rgba(255,255,255,0.95); border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); color: #333; }
                .summary { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin-bottom: 20px; backdrop-filter: blur(10px); }
                .metric { display: inline-block; margin: 10px 20px; font-size: 1.2em; }
                .metric-value { font-weight: bold; font-size: 1.5em; color: #4CAF50; }
            </style></head><body>
            <h1>🚀 NeuroGen-Alpha Cryptocurrency Trading Dashboard</h1>
            <div class="summary">
                <div class="metric">📊 Data Points: <span class="metric-value")" << data_series_.begin()->second.size() << R"(</span></div>
                <div class="metric">⏱️ Runtime: <span class="metric-value">)" << (data_series_.empty() ? 0 : data_series_.begin()->second.back().time) << R"( seconds</span></div>
                <div class="metric">🤖 Neural Network: <span class="metric-value">Active</span></div>
            </div>
            <div class="dashboard">)";

        for (const auto& pair : data_series_) {
            const std::string& name = pair.first;
            const auto& data = pair.second;
            
            if (data.empty()) continue;
            
            file << "<div class='chart' id='chart_" << name << "'></div>";
        }
        
        file << "</div><script>";
        
        for (const auto& pair : data_series_) {
            const std::string& name = pair.first;
            const auto& data = pair.second;
            
            if (data.empty()) continue;
            
            file << "var trace_" << name << " = { x: [";
            for(size_t i=0; i<data.size(); ++i) file << data[i].time << (i == data.size()-1 ? "" : ",");
            file << "], y: [";
            for(size_t i=0; i<data.size(); ++i) file << data[i].value << (i == data.size()-1 ? "" : ",");
            file << "], type: 'scatter', mode: 'lines+markers', name: '" << name << "', ";
            
            // Color coding based on series name
            if (name.find("Price") != std::string::npos) {
                file << "line: {color: '#FF6B6B', width: 3}";
            } else if (name.find("Portfolio") != std::string::npos) {
                file << "line: {color: '#4ECDC4', width: 3}";
            } else if (name.find("Volume") != std::string::npos) {
                file << "line: {color: '#45B7D1', width: 2}";
            } else {
                file << "line: {color: '#96CEB4', width: 2}";
            }
            
            file << "};";
            file << "var layout_" << name << " = { ";
            file << "title: {text: '" << name << "', font: {size: 18, color: '#333'}}, ";
            file << "xaxis: {title: 'Time (seconds)', gridcolor: '#ddd'}, ";
            file << "yaxis: {title: 'Value', gridcolor: '#ddd'}, ";
            file << "plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)', ";
            file << "margin: {l: 60, r: 30, t: 60, b: 50} };";
            file << "Plotly.newPlot('chart_" << name << "', [trace_" << name << "], layout_" << name << ");";
        }

        file << R"(
            // Auto-refresh every 30 seconds
            setInterval(function() {
                location.reload();
            }, 30000);
        </script></body></html>)";
        
        file.close();
        std::cout << "[CHARTS] Dashboard generated with " << data_series_.size() << " data series" << std::endl;
    }
};

// =============================================================================
// GLOBAL OBJECTS FOR SIGNAL HANDLING AND COORDINATION
// =============================================================================

std::unique_ptr<ChartGenerator> chart_generator_ptr;
std::unique_ptr<CryptoTradingSimulator> trading_simulator_ptr;
std::ofstream metrics_file;
std::atomic<bool> shutdown_requested(false);

// =============================================================================
// CRYPTOCURRENCY TRADING SIMULATOR
// =============================================================================

class CryptoTradingSimulator {
private:
    std::unique_ptr<KrakenAPIClient> api_client_;
    std::unique_ptr<TradingAgent> trading_agent_;
    std::unique_ptr<RealTimeMonitor> monitor_;
    
    std::string trading_pair_;
    int update_interval_seconds_;
    bool simulation_running_;
    
    std::thread trading_thread_;
    std::mutex simulation_mutex_;
    
public:
    CryptoTradingSimulator(const std::string& trading_pair, int update_interval = 60) 
        : trading_pair_(trading_pair)
        , update_interval_seconds_(update_interval)
        , simulation_running_(false)
    {
        // Initialize components
        api_client_ = std::make_unique<KrakenAPIClient>();
        trading_agent_ = std::make_unique<TradingAgent>(trading_pair);
        monitor_ = std::make_unique<RealTimeMonitor>();
        
        std::cout << "[SIMULATOR] Initialized for " << trading_pair_ << " with Kraken API" << std::endl;
    }
    
    ~CryptoTradingSimulator() {
        stopSimulation();
    }
    
    void initializeSystem() {
        std::cout << "[SIMULATOR] Initializing neural network and trading systems..." << std::endl;
        
        // Initialize neural network
        trading_agent_->initializeNeuralNetwork();
        
        // Load historical data for initialization using Kraken API
        try {
            std::cout << "[SIMULATOR] Loading historical market data from Kraken API..." << std::endl;
            auto historical_data = api_client_->getOHLCData(trading_pair_, 60, 50); // 50 hourly candles
            
            if (historical_data.empty()) {
                std::cout << "[WARNING] No historical data loaded from API, using synthetic data..." << std::endl;
                // Generate fallback synthetic data
                for (int i = 0; i < 10; ++i) {
                    auto synthetic_data = api_client_->generateFallbackData(trading_pair_);
                    trading_agent_->processMarketData(synthetic_data);
                    monitor_->updateMarketData(trading_pair_, synthetic_data);
                }
            } else {
                std::cout << "[SIMULATOR] Loaded " << historical_data.size() << " historical data points from Kraken" << std::endl;
                
                // Feed historical data to trading agent for initialization
                for (const auto& data : historical_data) {
                    if (data.validate()) {
                        trading_agent_->processMarketData(data);
                        monitor_->updateMarketData(trading_pair_, data);
                    } else {
                        std::cout << "[WARNING] Skipping invalid data point" << std::endl;
                    }
                }
            }
            
            std::cout << "[SIMULATOR] System initialized successfully with Kraken API" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to initialize with Kraken API: " << e.what() << std::endl;
            std::cout << "[SIMULATOR] Attempting to continue with synthetic data..." << std::endl;
            
            // Generate some synthetic data as fallback
            for (int i = 0; i < 10; ++i) {
                try {
                    auto synthetic_data = api_client_->generateFallbackData(trading_pair_);
                    trading_agent_->processMarketData(synthetic_data);
                    monitor_->updateMarketData(trading_pair_, synthetic_data);
                } catch (...) {
                    // If synthetic data also fails, continue anyway
                    break;
                }
            }
            
            std::cout << "[SIMULATOR] Fallback initialization completed" << std::endl;
        }
    }
    
    void startSimulation() {
        if (simulation_running_) {
            std::cout << "[SIMULATOR] Simulation already running" << std::endl;
            return;
        }
        
        std::cout << "[SIMULATOR] Starting real-time cryptocurrency trading simulation..." << std::endl;
        std::cout << "[SIMULATOR] Trading pair: " << trading_pair_ << std::endl;
        std::cout << "[SIMULATOR] Update interval: " << update_interval_seconds_ << " seconds" << std::endl;
        std::cout << "[SIMULATOR] Press Ctrl+C to stop gracefully" << std::endl;
        
        simulation_running_ = true;
        trading_thread_ = std::thread(&CryptoTradingSimulator::tradingLoop, this);
    }
    
    void stopSimulation() {
        if (!simulation_running_) {
            return;
        }
        
        std::cout << "[SIMULATOR] Stopping simulation..." << std::endl;
        simulation_running_ = false;
        
        if (trading_thread_.joinable()) {
            trading_thread_.join();
        }
        
        std::cout << "[SIMULATOR] Simulation stopped" << std::endl;
    }
    
    void printFinalReport() {
        std::cout << "\n[SIMULATOR] Generating final performance report..." << std::endl;
        
        if (trading_agent_) {
            trading_agent_->evaluatePerformance();
            trading_agent_->exportTradingLog("crypto_trading_log_" + trading_pair_ + ".csv");
            trading_agent_->exportPerformanceReport("crypto_performance_report_" + trading_pair_ + ".txt");
        }
    }
    
    TradingAgent* getTradingAgent() const { return trading_agent_.get(); }
    
private:
    void tradingLoop() {
        int cycle_count = 0;
        auto last_status_print = std::chrono::system_clock::now();
        
        while (simulation_running_ && !shutdown_requested.load()) {
            try {
                cycle_count++;
                auto cycle_start = std::chrono::system_clock::now();
                
                std::cout << "\n[CYCLE " << cycle_count << "] Fetching market data from Kraken API..." << std::endl;
                
                // Fetch latest market data from Kraken API
                auto latest_data = api_client_->getCurrentPrice(trading_pair_);
                
                std::cout << "[CYCLE " << cycle_count << "] " << trading_pair_ << " Price: $" 
                          << std::fixed << std::setprecision(2) << latest_data.close 
                          << " (Volume: " << latest_data.volume << ")" << std::endl;
                
                // Update monitor
                monitor_->updateMarketData(trading_pair_, latest_data);
                
                // Process data through trading agent
                trading_agent_->processMarketData(latest_data);
                
                // Log to metrics file
                if (metrics_file.is_open()) {
                    auto stats = trading_agent_->getStatistics();
                    metrics_file << latest_data.datetime << "," << trading_pair_ << ","
                                 << latest_data.close << "," << latest_data.volume << ","
                                 << stats.total_return << "," << stats.total_trades << "\n";
                    metrics_file.flush();
                }
                
                // Print status every 5 cycles
                auto now = std::chrono::system_clock::now();
                if (cycle_count % 5 == 0 || 
                    std::chrono::duration_cast<std::chrono::minutes>(now - last_status_print).count() >= 5) {
                    monitor_->printStatus(*trading_agent_, trading_pair_);
                    last_status_print = now;
                }
                
                // Chart generation for monitoring
                if (chart_generator_ptr) {
                    chart_generator_ptr->record("Price ($)", latest_data.close);
                    chart_generator_ptr->record("Volume", latest_data.volume);
                    chart_generator_ptr->record("Portfolio Value", trading_agent_->getStatistics().total_return);
                }
                
                // Calculate sleep time to maintain interval
                auto cycle_end = std::chrono::system_clock::now();
                auto cycle_duration = std::chrono::duration_cast<std::chrono::seconds>(cycle_end - cycle_start);
                int sleep_time = std::max(1, update_interval_seconds_ - static_cast<int>(cycle_duration.count()));
                
                std::cout << "[CYCLE " << cycle_count << "] Cycle completed in " 
                          << cycle_duration.count() << "s, sleeping for " << sleep_time << "s" << std::endl;
                
                // Sleep with periodic shutdown checks
                for (int i = 0; i < sleep_time && simulation_running_ && !shutdown_requested.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Trading loop error: " << e.what() << std::endl;
                std::cerr << "[ERROR] Retrying in 30 seconds..." << std::endl;
                
                // Error recovery - wait and retry
                for (int i = 0; i < 30 && simulation_running_ && !shutdown_requested.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
        }
        
        std::cout << "[SIMULATOR] Trading loop completed after " << cycle_count << " cycles" << std::endl;
    }
};

// =============================================================================
// SIGNAL HANDLING & MAIN APPLICATION
// =============================================================================

void handleSignal(int) {
    std::cout << "\n[SYSTEM] Shutdown signal received. Stopping gracefully..." << std::endl;
    shutdown_requested.store(true);
    
    if (trading_simulator_ptr) {
        trading_simulator_ptr->stopSimulation();
        trading_simulator_ptr->printFinalReport();
    }
    
    if (chart_generator_ptr) {
        chart_generator_ptr->generateDashboard("NeuroGen_Crypto_Dashboard_Final.html");
    }
    
    if (metrics_file.is_open()) {
        metrics_file.close();
    }
    
    std::cout << "[SYSTEM] Cleanup complete. Exiting..." << std::endl;
    cleanupNetwork();
    exit(0);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handleSignal); // Handle Ctrl+C
    
    try {
        std::cout << "🚀 NeuroGen-Alpha Cryptocurrency Trading System 🚀" << std::endl;
        std::cout << "======================================================" << std::endl;
        std::cout << "Neural Network + Real-Time Crypto Trading" << std::endl;
        std::cout << "======================================================" << std::endl;
        
        // Configuration
        std::string trading_pair = "BTCUSD";  // Default trading pair
        int update_interval = 60;              // Update every 60 seconds
        
        // Parse command line arguments
        if (argc > 1) {
            trading_pair = argv[1];
            std::cout << "[CONFIG] Trading pair set to: " << trading_pair << std::endl;
        }
        if (argc > 2) {
            update_interval = std::stoi(argv[2]);
            std::cout << "[CONFIG] Update interval set to: " << update_interval << " seconds" << std::endl;
        }
        
        // Initialize components
        chart_generator_ptr = std::make_unique<ChartGenerator>();
        
        // Initialize metrics logging
        std::string metrics_filename = "crypto_trading_metrics_" + trading_pair + ".csv";
        metrics_file.open(metrics_filename);
        metrics_file << "timestamp,symbol,price,volume,portfolio_return,total_trades\n";
        std::cout << "[SYSTEM] Metrics logging to: " << metrics_filename << std::endl;
        
        // Create and initialize trading simulator
        std::cout << "[SYSTEM] Initializing cryptocurrency trading simulator..." << std::endl;
        trading_simulator_ptr = std::make_unique<CryptoTradingSimulator>(trading_pair, update_interval);
        
        // Initialize the system with historical data and neural network
        trading_simulator_ptr->initializeSystem();
        
        std::cout << "\n[SYSTEM] Starting real-time trading simulation..." << std::endl;
        std::cout << "[SYSTEM] Trading: " << trading_pair << std::endl;
        std::cout << "[SYSTEM] Interval: " << update_interval << " seconds" << std::endl;
        std::cout << "[SYSTEM] Dashboard will be generated at: NeuroGen_Crypto_Dashboard_Final.html" << std::endl;
        std::cout << "[SYSTEM] Press Ctrl+C to stop and generate final reports" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Start the trading simulation
        trading_simulator_ptr->startSimulation();
        
        // Keep main thread alive and generate periodic dashboards
        auto last_dashboard_update = std::chrono::system_clock::now();
        while (!shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            auto now = std::chrono::system_clock::now();
            if (std::chrono::duration_cast<std::chrono::minutes>(now - last_dashboard_update).count() >= 5) {
                if (chart_generator_ptr) {
                    chart_generator_ptr->generateDashboard("NeuroGen_Crypto_Dashboard_Live.html");
                    last_dashboard_update = now;
                }
            }
        }
        
        // Graceful shutdown
        handleSignal(SIGINT);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        handleSignal(SIGINT);
        return 1;
    }
    
    return 0;
}