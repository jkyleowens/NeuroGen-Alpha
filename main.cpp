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
#include <chrono>
#include <cmath>

#if defined(USE_CUDA) && USE_CUDA
#include <cuda_runtime.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>
#else
#include "cpu_network_wrapper.h"
#endif

// Enhanced Portfolio Management System
class TradingPortfolio {
private:
    float cash_;
    float shares_;
    float last_price_;
    float initial_value_;
    std::vector<float> value_history_;
    std::vector<float> return_history_;
    int total_trades_;
    int winning_trades_;
    float max_drawdown_;
    float peak_value_;
    
public:
    TradingPortfolio(float initial_cash = 100000.0f) 
        : cash_(initial_cash), shares_(0.0f), last_price_(0.0f), 
          initial_value_(initial_cash), total_trades_(0), winning_trades_(0),
          max_drawdown_(0.0f), peak_value_(initial_cash) {
        value_history_.reserve(10000);
        return_history_.reserve(10000);
    }
    
    bool executeAction(const std::string& action, float current_price, float confidence = 1.0f) {
        if (current_price <= 0.0f) return false;
        
        float old_value = getTotalValue();
        bool trade_executed = false;
        
        if (action == "buy" && cash_ >= current_price && confidence > 0.6f) {
            // Position sizing based on confidence
            float position_size = std::min(cash_ / current_price, confidence * 10.0f);
            float shares_to_buy = std::floor(position_size);
            
            if (shares_to_buy >= 1.0f) {
                shares_ += shares_to_buy;
                cash_ -= shares_to_buy * current_price;
                total_trades_++;
                trade_executed = true;
            }
        } 
        else if (action == "sell" && shares_ >= 1.0f && confidence > 0.6f) {
            // Sell based on confidence (partial or full position)
            float shares_to_sell = std::min(shares_, confidence * shares_);
            shares_to_sell = std::floor(shares_to_sell);
            
            if (shares_to_sell >= 1.0f) {
                shares_ -= shares_to_sell;
                cash_ += shares_to_sell * current_price;
                total_trades_++;
                trade_executed = true;
                
                // Track winning trades
                if (current_price > last_price_) {
                    winning_trades_++;
                }
            }
        }
        
        last_price_ = current_price;
        
        // Update performance metrics
        float new_value = getTotalValue();
        value_history_.push_back(new_value);
        
        if (value_history_.size() > 1) {
            float return_pct = (new_value - old_value) / old_value;
            return_history_.push_back(return_pct);
        }
        
        // Update peak and drawdown
        if (new_value > peak_value_) {
            peak_value_ = new_value;
        } else {
            float current_drawdown = (peak_value_ - new_value) / peak_value_;
            max_drawdown_ = std::max(max_drawdown_, current_drawdown);
        }
        
        return trade_executed;
    }
    
    float computeReward() const {
        if (value_history_.size() < 2) return 0.0f;
        
        // Multi-factor reward signal
        float recent_return = (value_history_.back() - value_history_[value_history_.size()-2]) 
                             / value_history_[value_history_.size()-2];
        
        // Sharpe ratio component (simplified)
        float sharpe_component = 0.0f;
        if (return_history_.size() >= 10) {
            float mean_return = 0.0f;
            float return_variance = 0.0f;
            
            int lookback = std::min(50, static_cast<int>(return_history_.size()));
            for (int i = return_history_.size() - lookback; i < return_history_.size(); ++i) {
                mean_return += return_history_[i];
            }
            mean_return /= lookback;
            
            for (int i = return_history_.size() - lookback; i < return_history_.size(); ++i) {
                float diff = return_history_[i] - mean_return;
                return_variance += diff * diff;
            }
            return_variance /= lookback;
            
            if (return_variance > 1e-8f) {
                sharpe_component = mean_return / std::sqrt(return_variance);
            }
        }
        
        // Combined reward: recent performance + risk-adjusted performance
        float reward = recent_return * 10.0f + sharpe_component * 0.1f;
        
        // Penalty for excessive drawdown
        if (max_drawdown_ > 0.2f) {
            reward -= max_drawdown_ * 2.0f;
        }
        
        return std::tanh(reward); // Normalize to [-1, 1]
    }
    
    float getTotalValue() const {
        return cash_ + shares_ * last_price_;
    }
    
    float getReturnPercent() const {
        return (getTotalValue() - initial_value_) / initial_value_ * 100.0f;
    }
    
    float getSharpeRatio() const {
        if (return_history_.size() < 10) return 0.0f;
        
        float mean_return = 0.0f;
        float variance = 0.0f;
        
        for (float ret : return_history_) {
            mean_return += ret;
        }
        mean_return /= return_history_.size();
        
        for (float ret : return_history_) {
            float diff = ret - mean_return;
            variance += diff * diff;
        }
        variance /= return_history_.size();
        
        return variance > 1e-8f ? mean_return / std::sqrt(variance) : 0.0f;
    }
    
    void printSummary() const {
        std::cout << "\n=== Portfolio Performance Summary ===" << std::endl;
        std::cout << "Total Value: $" << std::fixed << std::setprecision(2) << getTotalValue() << std::endl;
        std::cout << "Return: " << std::setprecision(2) << getReturnPercent() << "%" << std::endl;
        std::cout << "Cash: $" << cash_ << ", Shares: " << shares_ << std::endl;
        std::cout << "Total Trades: " << total_trades_ << std::endl;
        std::cout << "Win Rate: " << (total_trades_ > 0 ? 100.0f * winning_trades_ / total_trades_ : 0.0f) << "%" << std::endl;
        std::cout << "Max Drawdown: " << std::setprecision(2) << max_drawdown_ * 100.0f << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << std::setprecision(3) << getSharpeRatio() << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    
    // Getters
    float getCash() const { return cash_; }
    float getShares() const { return shares_; }
    float getLastPrice() const { return last_price_; }
    int getTotalTrades() const { return total_trades_; }
    float getMaxDrawdown() const { return max_drawdown_; }
};

// Enhanced Feature Engineering
class FeatureEngineer {
private:
    std::vector<float> price_history_;
    std::vector<float> volume_history_;
    static constexpr int HISTORY_SIZE = 100;
    
public:
    std::vector<float> engineerFeatures(float open, float high, float low, float close, float volume) {
        // Update history
        price_history_.push_back(close);
        volume_history_.push_back(volume);
        
        if (price_history_.size() > HISTORY_SIZE) {
            price_history_.erase(price_history_.begin());
            volume_history_.erase(volume_history_.begin());
        }
        
        std::vector<float> features(60, 0.0f);
        
        // Basic OHLCV (normalized)
        float price_mean = close; // Simplified normalization base
        features[0] = open / price_mean - 1.0f;
        features[1] = high / price_mean - 1.0f;
        features[2] = low / price_mean - 1.0f;
        features[3] = close / price_mean - 1.0f;
        features[4] = std::log(volume + 1.0f) / 20.0f; // Log-normalized volume
        
        if (price_history_.size() >= 2) {
            // Price momentum indicators
            features[5] = (close - price_history_[price_history_.size()-2]) / price_mean;
            
            // Short-term moving averages
            if (price_history_.size() >= 5) {
                float ma5 = 0.0f;
                for (int i = price_history_.size() - 5; i < price_history_.size(); ++i) {
                    ma5 += price_history_[i];
                }
                ma5 /= 5.0f;
                features[6] = (close - ma5) / ma5;
            }
            
            if (price_history_.size() >= 20) {
                float ma20 = 0.0f;
                for (int i = price_history_.size() - 20; i < price_history_.size(); ++i) {
                    ma20 += price_history_[i];
                }
                ma20 /= 20.0f;
                features[7] = (close - ma20) / ma20;
                
                // Volatility (20-period)
                float variance = 0.0f;
                for (int i = price_history_.size() - 20; i < price_history_.size(); ++i) {
                    float diff = price_history_[i] - ma20;
                    variance += diff * diff;
                }
                variance /= 19.0f;
                features[8] = std::sqrt(variance) / ma20; // Relative volatility
            }
            
            // RSI-like momentum indicator
            if (price_history_.size() >= 14) {
                float gains = 0.0f, losses = 0.0f;
                for (int i = price_history_.size() - 14; i < price_history_.size() - 1; ++i) {
                    float change = price_history_[i+1] - price_history_[i];
                    if (change > 0) gains += change;
                    else losses += -change;
                }
                
                if (losses > 0) {
                    float rs = gains / losses;
                    features[9] = rs / (1.0f + rs); // RSI normalized to [0,1]
                } else {
                    features[9] = 1.0f;
                }
            }
        }
        
        // Volume indicators
        if (volume_history_.size() >= 10) {
            float vol_ma = 0.0f;
            for (int i = volume_history_.size() - 10; i < volume_history_.size(); ++i) {
                vol_ma += volume_history_[i];
            }
            vol_ma /= 10.0f;
            features[10] = (volume - vol_ma) / (vol_ma + 1.0f); // Relative volume
        }
        
        // Additional technical indicators would go in features[11] through features[59]
        // For now, they remain as padding zeros
        
        return features;
    }
};

// File I/O and Data Processing
std::vector<std::string> getAvailableDataFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    if (!std::filesystem::exists(directory)) {
        std::cerr << "[ERROR] Data directory '" << directory << "' does not exist." << std::endl;
        return files;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            files.push_back(entry.path().string());
        }
    }
    
    std::cout << "[INFO] Found " << files.size() << " CSV files in " << directory << std::endl;
    return files;
}

struct MarketData {
    float open, high, low, close, volume;
    std::string datetime;
    bool valid;
    
    MarketData() : open(0), high(0), low(0), close(0), volume(0), valid(false) {}
};

std::vector<MarketData> loadMarketData(const std::string& filepath) {
    std::vector<MarketData> data;
    std::ifstream file(filepath);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "[WARNING] Could not open: " << filepath << std::endl;
        return data;
    }
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> values;
        
        while (std::getline(ss, token, ',')) {
            values.push_back(token);
        }
        
        if (values.size() >= 6) {
            MarketData md;
            try {
                md.datetime = values[0];
                md.open = std::stof(values[1]);
                md.high = std::stof(values[2]);
                md.low = std::stof(values[3]);
                md.close = std::stof(values[4]);
                md.volume = std::stof(values[5]);
                md.valid = true;
                data.push_back(md);
            } catch (const std::exception& e) {
                // Skip invalid rows
                continue;
            }
        }
    }
    
    return data;
}

// Decision making with confidence
struct TradingDecision {
    std::string action;
    float confidence;
    std::vector<float> raw_outputs;
    
    TradingDecision(const std::vector<float>& outputs) : raw_outputs(outputs) {
        if (outputs.size() >= 3) {
            auto max_it = std::max_element(outputs.begin(), outputs.end());
            int max_idx = std::distance(outputs.begin(), max_it);
            confidence = *max_it;
            
            if (max_idx == 0) action = "buy";
            else if (max_idx == 1) action = "sell";
            else action = "hold";
        } else {
            action = "hold";
            confidence = 0.0f;
        }
    }
};

// Main Trading Simulation
int main(int argc, char* argv[]) {
    try {
        // Configuration
        std::string data_dir = "highly_diverse_stock_data";
        int num_epochs = 3;
        bool detailed_logging = false;
        
        if (argc > 1) data_dir = argv[1];
        if (argc > 2) num_epochs = std::stoi(argv[2]);
        if (argc > 3) detailed_logging = (std::string(argv[3]) == "verbose");
        
        std::cout << "=== Advanced Neural Trading Simulation ===" << std::endl;
        std::cout << "Data Directory: " << data_dir << std::endl;
        std::cout << "Epochs: " << num_epochs << std::endl;
        std::cout << "Detailed Logging: " << (detailed_logging ? "ON" : "OFF") << std::endl;
        std::cout << "===========================================" << std::endl;

        int device_count = 0;
#if defined(USE_CUDA) && USE_CUDA
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess || device_count == 0) {
            std::cerr << "[ERROR] No CUDA-capable device detected. Exiting." << std::endl;
            return 1;
        }
#else
        (void)device_count; // suppress unused warning
#endif
        
        // Load available data files
        auto data_files = getAvailableDataFiles(data_dir);
        if (data_files.empty()) {
            std::cerr << "[FATAL] No CSV files found in " << data_dir << std::endl;
            return 1;
        }
        
        // Initialize systems
        TradingPortfolio portfolio;
        FeatureEngineer feature_engineer;

        // Metrics logging
        std::ofstream metrics_file("network_metrics.csv");
        metrics_file << "epoch,portfolio_value,epoch_return,dopamine,neurons,synapses\n";
        float dopamine_level = 0.0f;
        
        // Initialize neural network on CUDA
        std::cout << "[INIT] Initializing neural network..." << std::endl;
        initializeNetwork();
        
        // Random number generation for file shuffling
        std::random_device rd;
        std::mt19937 rng(rd());
        
        long long total_decisions = 0;
        long long profitable_decisions = 0;
        auto simulation_start = std::chrono::high_resolution_clock::now();
        
        // Main training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n=== Epoch " << (epoch + 1) << "/" << num_epochs << " ===" << std::endl;
            
            auto epoch_start = std::chrono::high_resolution_clock::now();
            float epoch_start_value = portfolio.getTotalValue();
            
            // Shuffle files for better generalization
            std::shuffle(data_files.begin(), data_files.end(), rng);
            
            for (const auto& file_path : data_files) {
                std::filesystem::path p(file_path);
                
                if (detailed_logging) {
                    std::cout << "[PROCESSING] " << p.filename().string() << std::endl;
                }
                
                auto market_data = loadMarketData(file_path);
                if (market_data.empty()) continue;
                
                for (const auto& data_point : market_data) {
                    if (!data_point.valid) continue;
                    
                    // Engineer features
                    auto features = feature_engineer.engineerFeatures(
                        data_point.open, data_point.high, data_point.low, 
                        data_point.close, data_point.volume
                    );
                    
                    // Compute reward signal
                    float reward = portfolio.computeReward();
                    dopamine_level = 0.99f * dopamine_level + reward;
                    
                    // Neural network forward pass
                    auto start_forward = std::chrono::high_resolution_clock::now();
                    auto raw_outputs = forwardCUDA(features, reward);
                    auto end_forward = std::chrono::high_resolution_clock::now();
                    
                    // Make trading decision
                    TradingDecision decision(raw_outputs);
                    
                    // Execute trade
                    float old_value = portfolio.getTotalValue();
                    bool trade_executed = portfolio.executeAction(
                        decision.action, data_point.close, decision.confidence
                    );
                    float new_value = portfolio.getTotalValue();
                    
                    // Update neural network
                    auto start_learning = std::chrono::high_resolution_clock::now();
                    updateSynapticWeightsCUDA(reward);
                    auto end_learning = std::chrono::high_resolution_clock::now();
                    
                    total_decisions++;
                    if (new_value > old_value) profitable_decisions++;
                    
                    // Detailed logging
                    if (detailed_logging && total_decisions % 1000 == 0) {
                        float forward_time = std::chrono::duration<float, std::milli>(
                            end_forward - start_forward).count();
                        float learning_time = std::chrono::duration<float, std::milli>(
                            end_learning - start_learning).count();
                        
                        std::cout << "Decision " << total_decisions 
                                  << ": " << decision.action 
                                  << " (conf: " << std::fixed << std::setprecision(2) << decision.confidence << ")"
                                  << ", Price: $" << data_point.close
                                  << ", Value: $" << new_value
                                  << ", Reward: " << reward
                                  << ", Times: " << forward_time << "ms/" << learning_time << "ms" << std::endl;
                    }
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            float epoch_duration = std::chrono::duration<float>(epoch_end - epoch_start).count();
            float epoch_return = (portfolio.getTotalValue() - epoch_start_value) / epoch_start_value * 100.0f;
            
            std::cout << "Epoch " << (epoch + 1) << " completed in " << std::setprecision(1)
                      << epoch_duration << "s" << std::endl;
            std::cout << "Epoch Return: " << std::setprecision(2) << epoch_return << "%" << std::endl;
            std::cout << "Portfolio Value: $" << std::setprecision(2) << portfolio.getTotalValue() << std::endl;

            // Log metrics for this epoch
            NetworkStats stats = getNetworkStats();
            metrics_file << epoch << ',' << portfolio.getTotalValue() << ',' << epoch_return << ','
                         << dopamine_level << ',' << (stats.update_count > 0 ? stats.update_count : 0)
                         << ',' << getNetworkConfig().totalSynapses << '\n';
        }
        
        auto simulation_end = std::chrono::high_resolution_clock::now();
        float total_duration = std::chrono::duration<float>(simulation_end - simulation_start).count();
        
        // Final results
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "Total Duration: " << std::setprecision(1) << total_duration << " seconds" << std::endl;
        std::cout << "Total Decisions: " << total_decisions << std::endl;
        std::cout << "Profitable Decisions: " << profitable_decisions 
                  << " (" << (100.0f * profitable_decisions / total_decisions) << "%)" << std::endl;
        std::cout << "Decisions per Second: " << (total_decisions / total_duration) << std::endl;
        
        portfolio.printSummary();

        metrics_file.close();

        // Cleanup
        cleanupNetwork();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        cleanupNetwork();
        return 1;
    }
}
