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
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <cuda_runtime.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>

// Forward declarations for CUDA functions
extern "C" {
    void initializeNetwork();
    void cleanupNetwork();
    std::vector<float> forwardCUDA(const std::vector<float>& inputs, float reward);
    void updateSynapticWeightsCUDA(float reward);
}

// Global state variables
float dopamine_level = 0.5f;
std::ofstream metrics_file;

// Mathematical utility functions
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float tanh_activation(float x) { return std::tanh(x); }
inline float relu(float x) { return std::max(0.0f, x); }
inline float ema_alpha(int period) { return 2.0f / (period + 1.0f); }

// Advanced Technical Analysis Engine
class TechnicalAnalysis {
private:
    std::vector<float> prices_;
    std::vector<float> volumes_;
    std::vector<float> highs_;
    std::vector<float> lows_;
    std::vector<float> opens_;
    
    // EMA storage for efficiency
    float ema_12_ = 0.0f;
    float ema_26_ = 0.0f;
    float macd_ema_ = 0.0f;
    bool ema_initialized_ = false;
    
    static constexpr int MAX_HISTORY = 200;
    
public:
    struct TechnicalIndicators {
        float sma_5, sma_10, sma_20, sma_50;
        float ema_12, ema_26;
        float macd, macd_signal, macd_histogram;
        float rsi_14;
        float bb_upper, bb_middle, bb_lower, bb_squeeze;
        float atr_14;
        float stoch_k, stoch_d;
        float williams_r;
        float cci_20;
        float momentum_10;
        float roc_12;
        float volatility_20;
        float volume_sma_10, volume_ratio;
        bool valid = false;
    };
    
    void addData(float open, float high, float low, float close, float volume) {
        opens_.push_back(open);
        highs_.push_back(high);
        lows_.push_back(low);
        prices_.push_back(close);
        volumes_.push_back(volume);
        
        // Trim history to prevent memory bloat
        if (prices_.size() > MAX_HISTORY) {
            opens_.erase(opens_.begin());
            highs_.erase(highs_.begin());
            lows_.erase(lows_.begin());
            prices_.erase(prices_.begin());
            volumes_.erase(volumes_.begin());
        }
    }
    
    TechnicalIndicators calculateIndicators() {
        TechnicalIndicators indicators;
        
        if (prices_.empty()) return indicators;
        
        float current_price = prices_.back();
        int n = prices_.size();
        
        // Simple Moving Averages
        if (n >= 5) indicators.sma_5 = calculateSMA(5);
        if (n >= 10) indicators.sma_10 = calculateSMA(10);
        if (n >= 20) indicators.sma_20 = calculateSMA(20);
        if (n >= 50) indicators.sma_50 = calculateSMA(50);
        
        // Exponential Moving Averages and MACD
        if (n >= 26) {
            if (!ema_initialized_) {
                ema_12_ = calculateSMA(12);
                ema_26_ = calculateSMA(26);
                ema_initialized_ = true;
            } else {
                float alpha_12 = ema_alpha(12);
                float alpha_26 = ema_alpha(26);
                ema_12_ = alpha_12 * current_price + (1 - alpha_12) * ema_12_;
                ema_26_ = alpha_26 * current_price + (1 - alpha_26) * ema_26_;
            }
            
            indicators.ema_12 = ema_12_;
            indicators.ema_26 = ema_26_;
            indicators.macd = ema_12_ - ema_26_;
            
            // MACD Signal Line (EMA of MACD)
            float alpha_signal = ema_alpha(9);
            macd_ema_ = alpha_signal * indicators.macd + (1 - alpha_signal) * macd_ema_;
            indicators.macd_signal = macd_ema_;
            indicators.macd_histogram = indicators.macd - indicators.macd_signal;
        }
        
        // RSI
        if (n >= 14) indicators.rsi_14 = calculateRSI(14);
        
        // Bollinger Bands
        if (n >= 20) calculateBollingerBands(indicators);
        
        // Average True Range
        if (n >= 14) indicators.atr_14 = calculateATR(14);
        
        // Stochastic Oscillator
        if (n >= 14) calculateStochastic(indicators);
        
        // Williams %R
        if (n >= 14) indicators.williams_r = calculateWilliamsR(14);
        
        // Commodity Channel Index
        if (n >= 20) indicators.cci_20 = calculateCCI(20);
        
        // Momentum and Rate of Change
        if (n >= 10) indicators.momentum_10 = calculateMomentum(10);
        if (n >= 12) indicators.roc_12 = calculateROC(12);
        
        // Volatility
        if (n >= 20) indicators.volatility_20 = calculateVolatility(20);
        
        // Volume Analysis
        if (volumes_.size() >= 10) {
            indicators.volume_sma_10 = calculateVolumeSMA(10);
            indicators.volume_ratio = volumes_.back() / indicators.volume_sma_10;
        }
        
        indicators.valid = (n >= 20);
        return indicators;
    }
    
private:
    float calculateSMA(int period) {
        if (prices_.size() < period) return 0.0f;
        float sum = 0.0f;
        for (int i = prices_.size() - period; i < prices_.size(); ++i) {
            sum += prices_[i];
        }
        return sum / period;
    }
    
    float calculateRSI(int period) {
        if (prices_.size() <= period) return 50.0f;
        
        float gains = 0.0f, losses = 0.0f;
        for (int i = prices_.size() - period; i < prices_.size() - 1; ++i) {
            float change = prices_[i + 1] - prices_[i];
            if (change > 0) gains += change;
            else losses += -change;
        }
        
        float avg_gain = gains / period;
        float avg_loss = losses / period;
        
        if (avg_loss == 0.0f) return 100.0f;
        float rs = avg_gain / avg_loss;
        return 100.0f - (100.0f / (1.0f + rs));
    }
    
    void calculateBollingerBands(TechnicalIndicators& indicators) {
        float sma = calculateSMA(20);
        float variance = 0.0f;
        
        for (int i = prices_.size() - 20; i < prices_.size(); ++i) {
            float diff = prices_[i] - sma;
            variance += diff * diff;
        }
        
        float std_dev = std::sqrt(variance / 20.0f);
        indicators.bb_middle = sma;
        indicators.bb_upper = sma + 2.0f * std_dev;
        indicators.bb_lower = sma - 2.0f * std_dev;
        indicators.bb_squeeze = (indicators.bb_upper - indicators.bb_lower) / sma;
    }
    
    float calculateATR(int period) {
        if (highs_.size() < period) return 0.0f;
        
        float atr_sum = 0.0f;
        for (int i = std::max(1, (int)highs_.size() - period); i < highs_.size(); ++i) {
            float hl = highs_[i] - lows_[i];
            float hc = std::abs(highs_[i] - prices_[i-1]);
            float lc = std::abs(lows_[i] - prices_[i-1]);
            atr_sum += std::max({hl, hc, lc});
        }
        
        return atr_sum / std::min(period, (int)highs_.size() - 1);
    }
    
    void calculateStochastic(TechnicalIndicators& indicators) {
        if (highs_.size() < 14) return;
        
        float highest = *std::max_element(highs_.end() - 14, highs_.end());
        float lowest = *std::min_element(lows_.end() - 14, lows_.end());
        
        if (highest == lowest) {
            indicators.stoch_k = 50.0f;
        } else {
            indicators.stoch_k = 100.0f * (prices_.back() - lowest) / (highest - lowest);
        }
        
        // Simple %D (3-period SMA of %K) - simplified
        indicators.stoch_d = indicators.stoch_k; // Would need %K history for proper calculation
    }
    
    float calculateWilliamsR(int period) {
        if (highs_.size() < period) return 0.0f;
        
        float highest = *std::max_element(highs_.end() - period, highs_.end());
        float lowest = *std::min_element(lows_.end() - period, lows_.end());
        
        if (highest == lowest) return -50.0f;
        return -100.0f * (highest - prices_.back()) / (highest - lowest);
    }
    
    float calculateCCI(int period) {
        if (prices_.size() < period) return 0.0f;
        
        std::vector<float> typical_prices;
        for (int i = prices_.size() - period; i < prices_.size(); ++i) {
            typical_prices.push_back((highs_[i] + lows_[i] + prices_[i]) / 3.0f);
        }
        
        float sma_tp = std::accumulate(typical_prices.begin(), typical_prices.end(), 0.0f) / typical_prices.size();
        float mad = 0.0f;
        
        for (float tp : typical_prices) {
            mad += std::abs(tp - sma_tp);
        }
        mad /= typical_prices.size();
        
        if (mad == 0.0f) return 0.0f;
        return (typical_prices.back() - sma_tp) / (0.015f * mad);
    }
    
    float calculateMomentum(int period) {
        if (prices_.size() <= period) return 0.0f;
        return prices_.back() - prices_[prices_.size() - 1 - period];
    }
    
    float calculateROC(int period) {
        if (prices_.size() <= period) return 0.0f;
        float old_price = prices_[prices_.size() - 1 - period];
        if (old_price == 0.0f) return 0.0f;
        return ((prices_.back() - old_price) / old_price) * 100.0f;
    }
    
    float calculateVolatility(int period) {
        if (prices_.size() < period) return 0.0f;
        
        std::vector<float> returns;
        for (int i = prices_.size() - period; i < prices_.size() - 1; ++i) {
            if (prices_[i] != 0.0f) {
                returns.push_back(std::log(prices_[i + 1] / prices_[i]));
            }
        }
        
        if (returns.empty()) return 0.0f;
        
        float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
        float variance = 0.0f;
        
        for (float ret : returns) {
            float diff = ret - mean;
            variance += diff * diff;
        }
        
        return std::sqrt(variance / returns.size()) * std::sqrt(252.0f); // Annualized
    }
    
    float calculateVolumeSMA(int period) {
        if (volumes_.size() < period) return 0.0f;
        float sum = 0.0f;
        for (int i = volumes_.size() - period; i < volumes_.size(); ++i) {
            sum += volumes_[i];
        }
        return sum / period;
    }
};
// Enhanced Portfolio Management System with Risk Controls
class TradingPortfolio {
private:
    float cash_;
    float shares_;
    float last_price_;
    float initial_value_;
    std::vector<float> value_history_;
    std::vector<float> return_history_;
    std::vector<std::string> trade_log_;
    int total_trades_;
    int winning_trades_;
    float max_drawdown_;
    float peak_value_;
    float daily_var_95_;  // Value at Risk
    float position_limit_;  // Maximum position size
    
public:
    TradingPortfolio(float initial_cash = 100000.0f) 
        : cash_(initial_cash), shares_(0.0f), last_price_(0.0f), 
          initial_value_(initial_cash), total_trades_(0), winning_trades_(0),
          max_drawdown_(0.0f), peak_value_(initial_cash), daily_var_95_(0.0f),
          position_limit_(initial_cash * 0.95f) {  // 95% position limit
        value_history_.reserve(50000);
        return_history_.reserve(50000);
        trade_log_.reserve(10000);
    }
    
    bool executeAction(const std::string& action, float current_price, float confidence = 1.0f, 
                      const TechnicalAnalysis::TechnicalIndicators& indicators = {}) {
        if (current_price <= 0.0f) return false;
        
        float old_value = getTotalValue();
        bool trade_executed = false;
        
        // Risk management checks
        float max_position_value = position_limit_;
        float volatility_adjustment = indicators.valid ? std::max(0.1f, 1.0f - indicators.volatility_20 * 0.1f) : 1.0f;
        
        if (action == "buy" && cash_ >= current_price && confidence > 0.6f) {
            // Dynamic position sizing based on confidence, volatility, and technical indicators
            float base_position = std::min(cash_ / current_price, confidence * 20.0f);
            
            // Technical analysis adjustments
            float technical_multiplier = 1.0f;
            if (indicators.valid) {
                // Bullish technical signals increase position size
                if (indicators.rsi_14 < 30.0f) technical_multiplier += 0.2f;  // Oversold
                if (indicators.macd > indicators.macd_signal) technical_multiplier += 0.1f;  // MACD bullish
                if (current_price > indicators.bb_upper) technical_multiplier -= 0.2f;  // Overbought
                if (indicators.stoch_k < 20.0f) technical_multiplier += 0.15f;  // Stochastic oversold
            }
            
            float adjusted_position = base_position * technical_multiplier * volatility_adjustment;
            float shares_to_buy = std::floor(std::min(adjusted_position, max_position_value / current_price));
            
            if (shares_to_buy >= 1.0f) {
                shares_ += shares_to_buy;
                cash_ -= shares_to_buy * current_price;
                total_trades_++;
                trade_executed = true;
                
                // Log trade
                std::ostringstream log_entry;
                log_entry << "BUY " << shares_to_buy << " @ $" << std::fixed << std::setprecision(2) 
                         << current_price << " (conf: " << confidence << ", tech_mult: " << technical_multiplier << ")";
                trade_log_.push_back(log_entry.str());
            }
        } 
        else if (action == "sell" && shares_ >= 1.0f && confidence > 0.6f) {
            // Dynamic selling based on confidence and technical indicators
            float base_sell_ratio = confidence;
            
            // Technical analysis adjustments for selling
            float technical_multiplier = 1.0f;
            if (indicators.valid) {
                if (indicators.rsi_14 > 70.0f) technical_multiplier += 0.3f;  // Overbought - sell more
                if (indicators.macd < indicators.macd_signal) technical_multiplier += 0.2f;  // MACD bearish
                if (indicators.stoch_k > 80.0f) technical_multiplier += 0.2f;  // Stochastic overbought
                if (indicators.williams_r > -20.0f) technical_multiplier += 0.1f;  // Williams %R overbought
            }
            
            float sell_ratio = std::min(1.0f, base_sell_ratio * technical_multiplier);
            float shares_to_sell = std::floor(shares_ * sell_ratio);
            
            if (shares_to_sell >= 1.0f) {
                shares_ -= shares_to_sell;
                cash_ += shares_to_sell * current_price;
                total_trades_++;
                trade_executed = true;
                
                // Track winning trades
                if (current_price > last_price_) {
                    winning_trades_++;
                }
                
                // Log trade
                std::ostringstream log_entry;
                log_entry << "SELL " << shares_to_sell << " @ $" << std::fixed << std::setprecision(2) 
                         << current_price << " (conf: " << confidence << ", tech_mult: " << technical_multiplier << ")";
                trade_log_.push_back(log_entry.str());
            }
        }
        
        last_price_ = current_price;
        
        // Update performance metrics
        float new_value = getTotalValue();
        value_history_.push_back(new_value);
        
        if (value_history_.size() > 1) {
            float return_pct = (new_value - old_value) / old_value;
            return_history_.push_back(return_pct);
            
            // Update VaR calculation
            updateVaR();
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
    
private:
    void updateVaR() {
        if (return_history_.size() < 30) return;  // Need minimum data
        
        // Take last 100 returns for VaR calculation
        int lookback = std::min(100, static_cast<int>(return_history_.size()));
        std::vector<float> recent_returns(
            return_history_.end() - lookback, return_history_.end());
        
        std::sort(recent_returns.begin(), recent_returns.end());
        
        // 95% VaR (5th percentile)
        int var_index = static_cast<int>(recent_returns.size() * 0.05);
        daily_var_95_ = recent_returns[var_index] * getTotalValue();
    }
    
public:
    float computeReward() const {
        if (value_history_.size() < 2) return 0.0f;
        
        // Multi-factor reward signal with enhanced risk considerations
        float recent_return = (value_history_.back() - value_history_[value_history_.size()-2]) 
                             / value_history_[value_history_.size()-2];
        
        // Sharpe ratio component (risk-adjusted returns)
        float sharpe_component = 0.0f;
        if (return_history_.size() >= 20) {
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
        
        // Risk penalty components
        float drawdown_penalty = max_drawdown_ > 0.15f ? -max_drawdown_ * 3.0f : 0.0f;
        float var_penalty = daily_var_95_ < -getTotalValue() * 0.05f ? -0.5f : 0.0f;
        
        // Diversification bonus (simplified - based on cash/equity balance)
        float cash_ratio = cash_ / getTotalValue();
        float diversification_bonus = (cash_ratio > 0.1f && cash_ratio < 0.9f) ? 0.1f : 0.0f;
        
        // Combined reward: recent performance + risk-adjusted performance + risk penalties + diversification
        float reward = recent_return * 15.0f + sharpe_component * 0.2f + 
                      drawdown_penalty + var_penalty + diversification_bonus;
        
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
        
        float mean_return = std::accumulate(return_history_.begin(), return_history_.end(), 0.0f) / return_history_.size();
        
        float variance = 0.0f;
        for (float ret : return_history_) {
            float diff = ret - mean_return;
            variance += diff * diff;
        }
        variance /= return_history_.size();
        
        return variance > 1e-8f ? mean_return / std::sqrt(variance) : 0.0f;
    }
    
    float getInformationRatio() const {
        // Simplified IR assuming market return of 0.0004 (10% annual)
        float benchmark_return = 0.0004f;
        if (return_history_.size() < 10) return 0.0f;
        
        float mean_excess_return = 0.0f;
        float tracking_error = 0.0f;
        
        for (float ret : return_history_) {
            float excess = ret - benchmark_return;
            mean_excess_return += excess;
            tracking_error += excess * excess;
        }
        
        mean_excess_return /= return_history_.size();
        tracking_error = std::sqrt(tracking_error / return_history_.size());
        
        return tracking_error > 1e-8f ? mean_excess_return / tracking_error : 0.0f;
    }
    
    void printSummary() const {
        std::cout << "\n=== Enhanced Portfolio Performance Summary ===" << std::endl;
        std::cout << "Total Value: $" << std::fixed << std::setprecision(2) << getTotalValue() << std::endl;
        std::cout << "Return: " << std::setprecision(2) << getReturnPercent() << "%" << std::endl;
        std::cout << "Cash: $" << cash_ << ", Shares: " << shares_ << std::endl;
        std::cout << "Total Trades: " << total_trades_ << std::endl;
        std::cout << "Win Rate: " << (total_trades_ > 0 ? 100.0f * winning_trades_ / total_trades_ : 0.0f) << "%" << std::endl;
        std::cout << "Max Drawdown: " << std::setprecision(2) << max_drawdown_ * 100.0f << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << std::setprecision(3) << getSharpeRatio() << std::endl;
        std::cout << "Information Ratio: " << std::setprecision(3) << getInformationRatio() << std::endl;
        std::cout << "Daily VaR (95%): $" << std::setprecision(2) << daily_var_95_ << std::endl;
        
        if (trade_log_.size() > 0) {
            std::cout << "\n=== Recent Trades ===" << std::endl;
            int start_idx = std::max(0, static_cast<int>(trade_log_.size()) - 5);
            for (int i = start_idx; i < trade_log_.size(); ++i) {
                std::cout << trade_log_[i] << std::endl;
            }
        }
        std::cout << "=============================================" << std::endl;
    }
    
    // Getters
    float getCash() const { return cash_; }
    float getShares() const { return shares_; }
    float getLastPrice() const { return last_price_; }
    int getTotalTrades() const { return total_trades_; }
    float getMaxDrawdown() const { return max_drawdown_; }
    float getDailyVaR() const { return daily_var_95_; }
    const std::vector<float>& getValueHistory() const { return value_history_; }
    const std::vector<float>& getReturnHistory() const { return return_history_; }
};

// Advanced Neural Feature Engineering with 60+ Features
class AdvancedFeatureEngineer {
private:
    TechnicalAnalysis tech_analysis_;
    std::vector<float> price_history_;
    std::vector<float> volume_history_;
    std::vector<float> feature_history_;
    std::random_device rd_;
    std::mt19937 rng_;
    
    static constexpr int FEATURE_COUNT = 80;
    static constexpr int HISTORY_SIZE = 200;
    
public:
    AdvancedFeatureEngineer() : rng_(rd_()) {
        feature_history_.reserve(HISTORY_SIZE * FEATURE_COUNT);
    }
    
    std::vector<float> engineerFeatures(float open, float high, float low, float close, float volume) {
        // Update technical analysis engine
        tech_analysis_.addData(open, high, low, close, volume);
        auto indicators = tech_analysis_.calculateIndicators();
        
        // Update price/volume history
        price_history_.push_back(close);
        volume_history_.push_back(volume);
        
        if (price_history_.size() > HISTORY_SIZE) {
            price_history_.erase(price_history_.begin());
            volume_history_.erase(volume_history_.begin());
        }
        
        std::vector<float> features(FEATURE_COUNT, 0.0f);
        int idx = 0;
        
        // === CORE PRICE FEATURES (0-9) ===
        float price_norm = close > 0 ? close : 1.0f;
        features[idx++] = (open / price_norm) - 1.0f;      // 0: Normalized open
        features[idx++] = (high / price_norm) - 1.0f;      // 1: Normalized high  
        features[idx++] = (low / price_norm) - 1.0f;       // 2: Normalized low
        features[idx++] = 0.0f;                            // 3: Close (reference = 0)
        features[idx++] = std::tanh(std::log(volume + 1.0f) / 15.0f); // 4: Log-normalized volume
        
        // Price relationships
        float hl_ratio = (high > low) ? (close - low) / (high - low) : 0.5f;
        features[idx++] = hl_ratio;                        // 5: Close position in daily range
        features[idx++] = (high - low) / price_norm;       // 6: Daily range
        features[idx++] = (close - open) / price_norm;     // 7: Daily change
        features[idx++] = std::abs(close - open) / price_norm; // 8: Daily volatility
        features[idx++] = (volume > 0) ? (close - open) * volume / 1000000.0f : 0.0f; // 9: Volume-weighted change
        
        if (price_history_.size() >= 2) {
            float prev_close = price_history_[price_history_.size() - 2];
            
            // === MOMENTUM FEATURES (10-19) ===
            features[idx++] = (close - prev_close) / prev_close;  // 10: 1-period return
            
            // Multi-period momentum
            for (int period : {2, 3, 5, 10, 20}) {
                if (price_history_.size() > period) {
                    float old_price = price_history_[price_history_.size() - 1 - period];
                    features[idx++] = (close - old_price) / old_price;
                } else {
                    features[idx++] = 0.0f;
                }
            }
            
            // === MOVING AVERAGE FEATURES (15-24) ===
            if (indicators.valid) {
                features[idx++] = indicators.sma_5 > 0 ? (close - indicators.sma_5) / indicators.sma_5 : 0.0f;    // 15
                features[idx++] = indicators.sma_10 > 0 ? (close - indicators.sma_10) / indicators.sma_10 : 0.0f; // 16
                features[idx++] = indicators.sma_20 > 0 ? (close - indicators.sma_20) / indicators.sma_20 : 0.0f; // 17
                features[idx++] = indicators.sma_50 > 0 ? (close - indicators.sma_50) / indicators.sma_50 : 0.0f; // 18
                features[idx++] = (indicators.ema_12 - indicators.ema_26) / price_norm; // 19: MACD line
            } else {
                for (int i = 0; i < 5; ++i) features[idx++] = 0.0f;
            }
            
            // === TECHNICAL INDICATOR FEATURES (20-39) ===
            if (indicators.valid) {
                features[idx++] = indicators.rsi_14 / 100.0f - 0.5f;           // 20: RSI centered
                features[idx++] = std::tanh(indicators.macd / price_norm);      // 21: MACD normalized
                features[idx++] = std::tanh(indicators.macd_signal / price_norm); // 22: MACD signal
                features[idx++] = std::tanh(indicators.macd_histogram / price_norm); // 23: MACD histogram
                
                // Bollinger Bands
                float bb_position = 0.0f;
                if (indicators.bb_upper > indicators.bb_lower) {
                    bb_position = (close - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower) - 0.5f;
                }
                features[idx++] = bb_position;                                 // 24: BB position
                features[idx++] = std::tanh(indicators.bb_squeeze);            // 25: BB squeeze
                
                features[idx++] = indicators.atr_14 / price_norm;              // 26: ATR
                features[idx++] = indicators.stoch_k / 100.0f - 0.5f;          // 27: Stochastic %K
                features[idx++] = indicators.stoch_d / 100.0f - 0.5f;          // 28: Stochastic %D
                features[idx++] = (indicators.williams_r + 50.0f) / 100.0f;    // 29: Williams %R normalized
                features[idx++] = std::tanh(indicators.cci_20 / 100.0f);       // 30: CCI normalized
                features[idx++] = std::tanh(indicators.momentum_10 / price_norm); // 31: Momentum
                features[idx++] = std::tanh(indicators.roc_12 / 100.0f);       // 32: Rate of Change
                features[idx++] = std::tanh(indicators.volatility_20);         // 33: Volatility
                
                // Volume indicators
                features[idx++] = indicators.volume_sma_10 > 0 ? 
                    std::tanh(std::log(indicators.volume_ratio)) : 0.0f;       // 34: Volume ratio
                
                // Cross-indicator relationships
                features[idx++] = (indicators.rsi_14 > 70.0f) ? 1.0f : 
                                 (indicators.rsi_14 < 30.0f) ? -1.0f : 0.0f;   // 35: RSI signals
                features[idx++] = (indicators.macd > indicators.macd_signal) ? 1.0f : -1.0f; // 36: MACD signal cross
                features[idx++] = (close > indicators.bb_upper) ? 1.0f : 
                                 (close < indicators.bb_lower) ? -1.0f : 0.0f;  // 37: BB breakout
                features[idx++] = (indicators.stoch_k > 80.0f) ? 1.0f : 
                                 (indicators.stoch_k < 20.0f) ? -1.0f : 0.0f;   // 38: Stochastic signals
                features[idx++] = sigmoid(indicators.cci_20 / 100.0f);          // 39: CCI sigmoid
            } else {
                for (int i = 0; i < 20; ++i) features[idx++] = 0.0f;
            }
            
            // === ADVANCED PATTERN FEATURES (40-54) ===
            if (price_history_.size() >= 10) {
                // Trend strength (linear regression slope)
                float trend_slope = calculateTrendSlope(10);
                features[idx++] = std::tanh(trend_slope);                      // 40: 10-period trend
                
                // Support/Resistance levels
                auto [support, resistance] = calculateSupportResistance(20);
                features[idx++] = support > 0 ? (close - support) / support : 0.0f;     // 41: Distance to support
                features[idx++] = resistance > 0 ? (close - resistance) / resistance : 0.0f; // 42: Distance to resistance
                
                // Price pattern recognition
                features[idx++] = detectDoubleTop(10) ? 1.0f : 0.0f;          // 43: Double top pattern
                features[idx++] = detectDoubleBottom(10) ? 1.0f : 0.0f;       // 44: Double bottom pattern
                features[idx++] = calculateFractalDimension(20);               // 45: Market fractal dimension
                
                // Statistical features
                features[idx++] = calculateSkewness(20);                       // 46: Price skewness
                features[idx++] = calculateKurtosis(20);                       // 47: Price kurtosis
                features[idx++] = calculateHurstExponent(30);                  // 48: Hurst exponent
                
                // Market microstructure
                features[idx++] = calculateBidAskSpread();                     // 49: Simulated bid-ask spread
                features[idx++] = calculateMarketImpact(volume);               // 50: Market impact
                
                // Cyclical components
                features[idx++] = std::sin(2.0f * M_PI * (price_history_.size() % 252) / 252.0f); // 51: Annual cycle
                features[idx++] = std::cos(2.0f * M_PI * (price_history_.size() % 22) / 22.0f);   // 52: Monthly cycle
                features[idx++] = std::sin(2.0f * M_PI * (price_history_.size() % 5) / 5.0f);     // 53: Weekly cycle
                features[idx++] = calculateDominantCycle(50);                  // 54: Dominant price cycle
            } else {
                for (int i = 0; i < 15; ++i) features[idx++] = 0.0f;
            }
            
            // === VOLATILITY AND RISK FEATURES (55-64) ===
            if (price_history_.size() >= 20) {
                features[idx++] = calculateGARCHVolatility(20);                // 55: GARCH volatility
                features[idx++] = calculateParkinsonVolatility(20);            // 56: Parkinson volatility
                features[idx++] = calculateVaR(20, 0.05f);                     // 57: 5% VaR
                features[idx++] = calculateExpectedShortfall(20, 0.05f);       // 58: Expected shortfall
                features[idx++] = calculateMaxDrawdown(20);                    // 59: Rolling max drawdown
                features[idx++] = calculateVolatilityOfVolatility(30);         // 60: Vol of vol
                features[idx++] = calculateDownsideDeviation(20);              // 61: Downside deviation
                features[idx++] = calculateUpsideVolatility(20);               // 62: Upside volatility
                features[idx++] = calculateBeta(50);                           // 63: Market beta (simplified)
                features[idx++] = calculateJumpDetection();                    // 64: Jump detection
            } else {
                for (int i = 0; i < 10; ++i) features[idx++] = 0.0f;
            }
            
            // === MARKET REGIME FEATURES (65-74) ===
            features[idx++] = detectBullMarket(50) ? 1.0f : 0.0f;            // 65: Bull market regime
            features[idx++] = detectBearMarket(50) ? 1.0f : 0.0f;            // 66: Bear market regime
            features[idx++] = detectSidewaysMarket(30) ? 1.0f : 0.0f;        // 67: Sideways market
            features[idx++] = calculateMarketStress(20);                      // 68: Market stress level
            features[idx++] = calculateLiquidityScore(volume);                // 69: Liquidity score
            features[idx++] = calculateMomentumRegime(20);                    // 70: Momentum regime
            features[idx++] = calculateMeanReversionScore(15);                // 71: Mean reversion tendency
            features[idx++] = calculateTrendStrength(25);                     // 72: Trend strength
            features[idx++] = calculateMarketEfficiency(30);                  // 73: Market efficiency
            features[idx++] = calculateNoiseRatio(20);                        // 74: Signal-to-noise ratio
            
            // === ENSEMBLE AND META FEATURES (75-79) ===
            features[idx++] = calculateFeatureInteraction(features);          // 75: Feature interaction
            features[idx++] = calculateEnsembleSignal(features);              // 76: Ensemble signal
            features[idx++] = addControlledNoise();                           // 77: Controlled noise for regularization
            features[idx++] = calculateMarketSentiment(features);             // 78: Derived market sentiment
            features[idx++] = calculateComplexityScore(features);             // 79: Market complexity score
        }
        
        // Store feature history for time-series analysis
        feature_history_.insert(feature_history_.end(), features.begin(), features.end());
        if (feature_history_.size() > HISTORY_SIZE * FEATURE_COUNT) {
            feature_history_.erase(feature_history_.begin(), 
                                 feature_history_.begin() + FEATURE_COUNT);
        }
        
        return features;
    }

private:
    // Helper methods for advanced feature calculation
    float calculateTrendSlope(int period) {
        if (price_history_.size() < period) return 0.0f;
        
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        for (int i = 0; i < period; ++i) {
            float x = i;
            float y = price_history_[price_history_.size() - period + i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        float n = period;
        float denominator = n * sum_x2 - sum_x * sum_x;
        return (denominator != 0) ? (n * sum_xy - sum_x * sum_y) / denominator : 0.0f;
    }
    
    std::pair<float, float> calculateSupportResistance(int period) {
        if (price_history_.size() < period) return {0.0f, 0.0f};
        
        auto start_it = price_history_.end() - period;
        float support = *std::min_element(start_it, price_history_.end());
        float resistance = *std::max_element(start_it, price_history_.end());
        
        return {support, resistance};
    }
    
    bool detectDoubleTop(int lookback) {
        if (price_history_.size() < lookback) return false;
        // Simplified double top detection
        auto start_it = price_history_.end() - lookback;
        auto max_it = std::max_element(start_it, price_history_.end());
        return (max_it != price_history_.end() - 1) && 
               (*max_it > price_history_.back() * 1.02f);
    }
    
    bool detectDoubleBottom(int lookback) {
        if (price_history_.size() < lookback) return false;
        auto start_it = price_history_.end() - lookback;
        auto min_it = std::min_element(start_it, price_history_.end());
        return (min_it != price_history_.end() - 1) && 
               (*min_it < price_history_.back() * 0.98f);
    }
    
    float calculateFractalDimension(int period) {
        // Simplified Hurst exponent estimation
        return calculateHurstExponent(period);
    }
    
    float calculateSkewness(int period) {
        if (price_history_.size() < period) return 0.0f;
        
        std::vector<float> returns;
        for (int i = price_history_.size() - period; i < price_history_.size() - 1; ++i) {
            returns.push_back(std::log(price_history_[i + 1] / price_history_[i]));
        }
        
        float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
        float variance = 0.0f, skewness = 0.0f;
        
        for (float ret : returns) {
            float diff = ret - mean;
            variance += diff * diff;
            skewness += diff * diff * diff;
        }
        
        variance /= returns.size();
        skewness /= returns.size();
        
        float std_dev = std::sqrt(variance);
        return (std_dev > 1e-8f) ? skewness / (std_dev * std_dev * std_dev) : 0.0f;
    }
    
    float calculateKurtosis(int period) {
        if (price_history_.size() < period) return 3.0f; // Normal distribution kurtosis
        
        std::vector<float> returns;
        for (int i = price_history_.size() - period; i < price_history_.size() - 1; ++i) {
            returns.push_back(std::log(price_history_[i + 1] / price_history_[i]));
        }
        
        float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
        float variance = 0.0f, kurtosis = 0.0f;
        
        for (float ret : returns) {
            float diff = ret - mean;
            variance += diff * diff;
            kurtosis += diff * diff * diff * diff;
        }
        
        variance /= returns.size();
        kurtosis /= returns.size();
        
        return (variance > 1e-8f) ? (kurtosis / (variance * variance)) - 3.0f : 0.0f;
    }
    
    float calculateHurstExponent(int period) {
        if (price_history_.size() < period) return 0.5f;
        
        // Simplified R/S analysis
        std::vector<float> log_prices;
        for (int i = price_history_.size() - period; i < price_history_.size(); ++i) {
            log_prices.push_back(std::log(price_history_[i]));
        }
        
        float mean_log = std::accumulate(log_prices.begin(), log_prices.end(), 0.0f) / log_prices.size();
        
        float range = 0.0f, std_dev = 0.0f;
        float cumulative_dev = 0.0f;
        float min_dev = 0.0f, max_dev = 0.0f;
        
        for (int i = 0; i < log_prices.size(); ++i) {
            float dev = log_prices[i] - mean_log;
            cumulative_dev += dev;
            min_dev = std::min(min_dev, cumulative_dev);
            max_dev = std::max(max_dev, cumulative_dev);
            std_dev += dev * dev;
        }
        
        range = max_dev - min_dev;
        std_dev = std::sqrt(std_dev / log_prices.size());
        
        float rs_ratio = (std_dev > 1e-8f) ? range / std_dev : 1.0f;
        return (rs_ratio > 1e-8f) ? std::log(rs_ratio) / std::log(period) : 0.5f;
    }
    
    // Additional helper methods (simplified implementations)
    float calculateBidAskSpread() { return 0.001f; } // Simplified
    float calculateMarketImpact(float volume) { return std::tanh(volume / 1000000.0f) * 0.01f; }
    float calculateDominantCycle(int period) { return 0.5f; } // Placeholder
    float calculateGARCHVolatility(int period) { return calculateVolatility(period); }
    float calculateParkinsonVolatility(int period) { return calculateVolatility(period) * 1.2f; }
    float calculateVaR(int period, float confidence) { return -0.05f; } // Simplified
    float calculateExpectedShortfall(int period, float confidence) { return -0.07f; }
    float calculateMaxDrawdown(int period) { return 0.1f; }
    float calculateVolatilityOfVolatility(int period) { return 0.3f; }
    float calculateDownsideDeviation(int period) { return calculateVolatility(period) * 0.8f; }
    float calculateUpsideVolatility(int period) { return calculateVolatility(period) * 1.1f; }
    float calculateBeta(int period) { return 1.0f; } // Market beta
    float calculateJumpDetection() { return 0.0f; }
    
    float calculateVolatility(int period) {
        if (price_history_.size() < period) return 0.0f;
        
        std::vector<float> returns;
        for (int i = price_history_.size() - period; i < price_history_.size() - 1; ++i) {
            returns.push_back(std::log(price_history_[i + 1] / price_history_[i]));
        }
        
        float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
        float variance = 0.0f;
        
        for (float ret : returns) {
            float diff = ret - mean;
            variance += diff * diff;
        }
        
        return std::sqrt(variance / returns.size());
    }
    
    bool detectBullMarket(int period) { return calculateTrendSlope(period) > 0.001f; }
    bool detectBearMarket(int period) { return calculateTrendSlope(period) < -0.001f; }
    bool detectSidewaysMarket(int period) { return std::abs(calculateTrendSlope(period)) < 0.0005f; }
    float calculateMarketStress(int period) { return std::min(1.0f, calculateVolatility(period) * 10.0f); }
    float calculateLiquidityScore(float volume) { return sigmoid(volume / 1000000.0f); }
    float calculateMomentumRegime(int period) { return std::tanh(calculateTrendSlope(period) * 100.0f); }
    float calculateMeanReversionScore(int period) { return 1.0f - std::abs(calculateTrendSlope(period) * 50.0f); }
    float calculateTrendStrength(int period) { return std::abs(calculateTrendSlope(period)) * 100.0f; }
    float calculateMarketEfficiency(int period) { return sigmoid(calculateHurstExponent(period) - 0.5f); }
    float calculateNoiseRatio(int period) { return calculateVolatility(period) / (std::abs(calculateTrendSlope(period)) + 1e-8f); }
    
    float calculateFeatureInteraction(const std::vector<float>& features) {
        return features.size() > 10 ? features[0] * features[5] + features[1] * features[6] : 0.0f;
    }
    
    float calculateEnsembleSignal(const std::vector<float>& features) {
        float signal = 0.0f;
        for (int i = 0; i < std::min(20, (int)features.size()); ++i) {
            signal += features[i] * (i % 2 == 0 ? 1.0f : -1.0f);
        }
        return std::tanh(signal / 10.0f);
    }
    
    float addControlledNoise() {
        std::normal_distribution<float> noise(0.0f, 0.01f);
        return noise(rng_);
    }
    
    float calculateMarketSentiment(const std::vector<float>& features) {
        if (features.size() < 10) return 0.0f;
        return std::tanh((features[0] + features[1] + features[2] - features[3] - features[4]) / 5.0f);
    }
    
    float calculateComplexityScore(const std::vector<float>& features) {
        float complexity = 0.0f;
        for (float f : features) {
            complexity += std::abs(f);
        }
        return std::tanh(complexity / features.size());
    }
};

// Advanced Chart Generation System
class ChartGenerator {
private:
    struct DataPoint {
        float value;
        std::string label;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    
    std::vector<DataPoint> portfolio_values_;
    std::vector<DataPoint> neural_activity_;
    std::vector<DataPoint> reward_signals_;
    std::vector<DataPoint> synaptic_weights_;
    std::vector<DataPoint> neurotransmitter_levels_;
    std::vector<DataPoint> learning_rates_;
    
    static constexpr int MAX_CHART_POINTS = 10000;
    
public:
    void recordPortfolioValue(float value) {
        addDataPoint(portfolio_values_, value, "Portfolio");
    }
    
    void recordNeuralActivity(float activity) {
        addDataPoint(neural_activity_, activity, "Neural");
    }
    
    void recordRewardSignal(float reward) {
        addDataPoint(reward_signals_, reward, "Reward");
    }
    
    void recordSynapticWeight(float weight) {
        addDataPoint(synaptic_weights_, weight, "Synaptic");
    }
    
    void recordNeurotransmitter(float level) {
        addDataPoint(neurotransmitter_levels_, level, "Dopamine");
    }
    
    void recordLearningRate(float rate) {
        addDataPoint(learning_rates_, rate, "Learning");
    }
    
    void generateAllCharts(const std::string& output_dir = "charts") {
        // Create output directory
        std::filesystem::create_directories(output_dir);
        
        std::cout << "\n=== Generating Performance Charts ===" << std::endl;
        
        generatePortfolioChart(output_dir + "/portfolio_performance.html");
        generateNeuralActivityChart(output_dir + "/neural_activity.html");
        generateRewardChart(output_dir + "/reward_signals.html");
        generateSynapticChart(output_dir + "/synaptic_weights.html");
        generateNeurotransmitterChart(output_dir + "/neurotransmitter_levels.html");
        generateLearningChart(output_dir + "/learning_rates.html");
        generateCombinedDashboard(output_dir + "/dashboard.html");
        
        std::cout << "Charts generated in: " << output_dir << std::endl;
        std::cout << "Open dashboard.html for comprehensive view" << std::endl;
    }
    
private:
    void addDataPoint(std::vector<DataPoint>& container, float value, const std::string& label) {
        container.push_back({value, label, std::chrono::high_resolution_clock::now()});
        
        if (container.size() > MAX_CHART_POINTS) {
            container.erase(container.begin());
        }
    }
    
    void generatePortfolioChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Performance</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Portfolio Performance Analysis</h1>
    <div id="portfolioChart" class="chart"></div>
    <script>
        var portfolioData = [{
            x: [)";
        
        // Generate time series data
        for (size_t i = 0; i < portfolio_values_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(portfolio_values_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < portfolio_values_.size(); ++i) {
            if (i > 0) file << ", ";
            file << portfolio_values_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio Value',
            line: { color: 'rgb(0, 100, 200)', width: 2 }
        }];
        
        var layout = {
            title: 'Portfolio Value Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Value ($)' },
            showlegend: true
        };
        
        Plotly.newPlot('portfolioChart', portfolioData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateNeuralActivityChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Activity</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Neural Network Activity Analysis</h1>
    <div id="neuralChart" class="chart"></div>
    <script>
        var neuralData = [{
            x: [)";
        
        for (size_t i = 0; i < neural_activity_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(neural_activity_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < neural_activity_.size(); ++i) {
            if (i > 0) file << ", ";
            file << neural_activity_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Neural Activity',
            line: { color: 'rgb(200, 50, 50)', width: 2 }
        }];
        
        var layout = {
            title: 'Neural Network Activity Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Activity Level' },
            showlegend: true
        };
        
        Plotly.newPlot('neuralChart', neuralData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateRewardChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Reward Signals</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Reward Signal Analysis</h1>
    <div id="rewardChart" class="chart"></div>
    <script>
        var rewardData = [{
            x: [)";
        
        for (size_t i = 0; i < reward_signals_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(reward_signals_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < reward_signals_.size(); ++i) {
            if (i > 0) file << ", ";
            file << reward_signals_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Reward Signal',
            line: { color: 'rgb(50, 200, 50)', width: 2 }
        }];
        
        var layout = {
            title: 'Reward Signals Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Reward Value' },
            showlegend: true
        };
        
        Plotly.newPlot('rewardChart', rewardData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateSynapticChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Synaptic Weight Evolution</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Synaptic Weight Evolution</h1>
    <div id="synapticChart" class="chart"></div>
    <script>
        var synapticData = [{
            x: [)";
        
        for (size_t i = 0; i < synaptic_weights_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(synaptic_weights_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < synaptic_weights_.size(); ++i) {
            if (i > 0) file << ", ";
            file << synaptic_weights_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Synaptic Weights',
            line: { color: 'rgb(150, 50, 200)', width: 2 }
        }];
        
        var layout = {
            title: 'Synaptic Weight Changes Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Weight Value' },
            showlegend: true
        };
        
        Plotly.newPlot('synapticChart', synapticData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateNeurotransmitterChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Neurotransmitter Levels</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Neurotransmitter Level Analysis</h1>
    <div id="neurotransmitterChart" class="chart"></div>
    <script>
        var neurotransmitterData = [{
            x: [)";
        
        for (size_t i = 0; i < neurotransmitter_levels_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(neurotransmitter_levels_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < neurotransmitter_levels_.size(); ++i) {
            if (i > 0) file << ", ";
            file << neurotransmitter_levels_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Dopamine Level',
            line: { color: 'rgb(255, 165, 0)', width: 2 }
        }];
        
        var layout = {
            title: 'Neurotransmitter Levels Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Concentration Level' },
            showlegend: true
        };
        
        Plotly.newPlot('neurotransmitterChart', neurotransmitterData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateLearningChart(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Learning Rate Adaptation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Learning Rate Adaptation</h1>
    <div id="learningChart" class="chart"></div>
    <script>
        var learningData = [{
            x: [)";
        
        for (size_t i = 0; i < learning_rates_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(learning_rates_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < learning_rates_.size(); ++i) {
            if (i > 0) file << ", ";
            file << learning_rates_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            name: 'Learning Rate',
            line: { color: 'rgb(100, 150, 50)', width: 2 }
        }];
        
        var layout = {
            title: 'Learning Rate Changes Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Learning Rate' },
            showlegend: true
        };
        
        Plotly.newPlot('learningChart', learningData, layout);
    </script>
</body>
</html>)";
        file.close();
    }
    
    void generateCombinedDashboard(const std::string& filename) {
        std::ofstream file(filename);
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>Neural Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .full-width { grid-column: 1 / -1; }
        h1 { text-align: center; color: #333; }
        h2 { color: #666; margin-top: 0; }
    </style>
</head>
<body>
    <h1>Neural Trading System Dashboard</h1>
    <div class="dashboard">
        <div class="chart full-width">
            <h2>Portfolio Performance</h2>
            <div id="portfolioChart"></div>
        </div>
        <div class="chart">
            <h2>Neural Activity</h2>
            <div id="neuralChart"></div>
        </div>
        <div class="chart">
            <h2>Reward Signals</h2>
            <div id="rewardChart"></div>
        </div>
        <div class="chart">
            <h2>Synaptic Weights</h2>
            <div id="synapticChart"></div>
        </div>
        <div class="chart">
            <h2>Neurotransmitter Levels</h2>
            <div id="neurotransmitterChart"></div>
        </div>
    </div>
    
    <script>
        // Portfolio Chart Data
        var portfolioData = [{
            x: [)";
        
        for (size_t i = 0; i < portfolio_values_.size(); ++i) {
            if (i > 0) file << ", ";
            file << "'" << formatTimestamp(portfolio_values_[i].timestamp) << "'";
        }
        
        file << R"(],
            y: [)";
            
        for (size_t i = 0; i < portfolio_values_.size(); ++i) {
            if (i > 0) file << ", ";
            file << portfolio_values_[i].value;
        }
        
        file << R"(],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgb(0, 100, 200)', width: 3 }
        }];
        
        Plotly.newPlot('portfolioChart', portfolioData, {
            margin: { t: 30, b: 40, l: 50, r: 20 },
            showlegend: false
        });
        
        // Add other charts with similar data...
        // (Neural, Reward, Synaptic, Neurotransmitter charts would follow similar pattern)
    </script>
</body>
</html>)";
        file.close();
    }
    
    std::string formatTimestamp(const std::chrono::time_point<std::chrono::high_resolution_clock>& tp) {
        auto time_t = std::chrono::system_clock::to_time_t(
            std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                tp - std::chrono::high_resolution_clock::now() + std::chrono::system_clock::now()));
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        return oss.str();
    }
};
// Enhanced File I/O and Data Processing with Randomized Time Series
std::vector<std::string> getAvailableDataFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    if (!std::filesystem::exists(directory)) {
        std::cerr << "[WARNING] Data directory '" << directory << "' does not exist." << std::endl;
        std::cerr << "[INFO] Generating synthetic market data..." << std::endl;
        generateSyntheticData(directory);
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            files.push_back(entry.path().string());
        }
    }
    
    std::cout << "[INFO] Found " << files.size() << " CSV files in " << directory << std::endl;
    return files;
}

void generateSyntheticData(const std::string& directory) {
    std::filesystem::create_directories(directory);
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> returns(-0.0001f, 0.02f);  // Slight negative drift with 2% volatility
    std::uniform_real_distribution<float> volume_dist(500000.0f, 5000000.0f);
    
    std::vector<std::string> tickers = {"BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "AVAXUSD", "LINKUSD", "MATICUSD"};
    
    for (const auto& ticker : tickers) {
        std::string filename = directory + "/" + ticker + "_synthetic.csv";
        std::ofstream file(filename);
        
        file << "timestamp,open,high,low,close,volume\n";
        
        float price = 100.0f + rng() % 10000;  // Random starting price
        
        for (int i = 0; i < 10000; ++i) {  // Generate 10k data points per asset
            float ret = returns(rng);
            float new_price = price * (1.0f + ret);
            
            // Generate OHLC with realistic relationships
            float volatility = std::abs(ret) * price * 2.0f;
            std::uniform_real_distribution<float> ohlc_dist(-volatility, volatility);
            
            float open = price + ohlc_dist(rng) * 0.3f;
            float high = std::max({open, new_price}) + std::abs(ohlc_dist(rng));
            float low = std::min({open, new_price}) - std::abs(ohlc_dist(rng));
            float close = new_price;
            float volume = volume_dist(rng);
            
            // Ensure realistic OHLC relationships
            high = std::max({open, high, low, close});
            low = std::min({open, high, low, close});
            
            auto now = std::chrono::system_clock::now();
            auto timestamp = now - std::chrono::minutes(10000 - i);
            auto time_t = std::chrono::system_clock::to_time_t(timestamp);
            
            file << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                 << std::fixed << std::setprecision(6)
                 << open << "," << high << "," << low << "," << close << ","
                 << static_cast<long>(volume) << "\n";
            
            price = new_price;
        }
        
        file.close();
        std::cout << "[GENERATED] " << filename << " with 10k data points" << std::endl;
    }
}

// Enhanced Market Data Structure with Validation
struct MarketData {
    float open, high, low, close, volume;
    std::string datetime;
    bool valid;
    float true_range;  // For ATR calculation
    float typical_price;  // (H+L+C)/3
    
    MarketData() : open(0), high(0), low(0), close(0), volume(0), valid(false), true_range(0), typical_price(0) {}
    
    void calculateDerivedFields(float prev_close = 0.0f) {
        if (valid) {
            // True Range calculation
            float hl = high - low;
            float hc = (prev_close > 0) ? std::abs(high - prev_close) : hl;
            float lc = (prev_close > 0) ? std::abs(low - prev_close) : hl;
            true_range = std::max({hl, hc, lc});
            
            // Typical Price
            typical_price = (high + low + close) / 3.0f;
        }
    }
    
    bool validate() {
        // Enhanced validation
        if (open <= 0 || high <= 0 || low <= 0 || close <= 0 || volume < 0) {
            valid = false;
            return false;
        }
        
        // OHLC relationship checks
        if (high < std::max({open, close}) || low > std::min({open, close})) {
            valid = false;
            return false;
        }
        
        // Reasonable price movement check (reject obvious outliers)
        float max_price = std::max({open, high, low, close});
        float min_price = std::min({open, high, low, close});
        if (max_price > min_price * 10.0f) {  // 1000% move in single period
            valid = false;
            return false;
        }
        
        valid = true;
        return true;
    }
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
    
    float prev_close = 0.0f;
    int valid_count = 0, invalid_count = 0;
    
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
                
                if (md.validate()) {
                    md.calculateDerivedFields(prev_close);
                    data.push_back(md);
                    prev_close = md.close;
                    valid_count++;
                } else {
                    invalid_count++;
                }
            } catch (const std::exception& e) {
                invalid_count++;
                continue;
            }
        }
    }
    
    std::cout << "[LOADED] " << filepath << ": " << valid_count << " valid, " 
              << invalid_count << " invalid records" << std::endl;
    
    return data;
}

// Load every CSV file once before training to avoid repeated I/O overhead
std::vector<std::vector<MarketData>> preloadMarketData(
    const std::vector<std::string>& files) {
    std::vector<std::vector<MarketData>> all;
    all.reserve(files.size());
    for (const auto& f : files) {
        all.push_back(loadMarketData(f));
    }
    return all;
}

// Enhanced Trading Decision with Multi-Factor Analysis
struct TradingDecision {
    std::string action;
    float confidence;
    float technical_score;
    float neural_score;
    float risk_score;
    std::vector<float> raw_outputs;
    std::string reasoning;
    
    TradingDecision(const std::vector<float>& outputs, 
                   const TechnicalAnalysis::TechnicalIndicators& indicators = {}) 
        : raw_outputs(outputs), technical_score(0.0f), neural_score(0.0f), risk_score(0.0f) {
        
        if (outputs.size() >= 3) {
            // Neural network decision
            auto max_it = std::max_element(outputs.begin(), outputs.end());
            int max_idx = std::distance(outputs.begin(), max_it);
            neural_score = *max_it;
            
            // Technical analysis confirmation
            if (indicators.valid) {
                calculateTechnicalScore(indicators);
                calculateRiskScore(indicators);
            }
            
            // Combined confidence
            confidence = (neural_score * 0.6f + technical_score * 0.3f + (1.0f - risk_score) * 0.1f);
            confidence = std::clamp(confidence, 0.0f, 1.0f);
            
            // Final decision with reasoning
            if (max_idx == 0 && confidence > 0.65f) {
                action = "buy";
                reasoning = generateBuyReasoning(indicators);
            } else if (max_idx == 1 && confidence > 0.65f) {
                action = "sell";
                reasoning = generateSellReasoning(indicators);
            } else {
                action = "hold";
                reasoning = "Insufficient confidence or mixed signals";
            }
        } else {
            action = "hold";
            confidence = 0.0f;
            reasoning = "Invalid neural network output";
        }
    }

private:
    void calculateTechnicalScore(const TechnicalAnalysis::TechnicalIndicators& indicators) {
        float bullish_signals = 0.0f;
        float bearish_signals = 0.0f;
        int signal_count = 0;
        
        // RSI signals
        if (indicators.rsi_14 < 30.0f) bullish_signals += 1.0f;
        else if (indicators.rsi_14 > 70.0f) bearish_signals += 1.0f;
        signal_count++;
        
        // MACD signals
        if (indicators.macd > indicators.macd_signal) bullish_signals += 1.0f;
        else bearish_signals += 1.0f;
        signal_count++;
        
        // Stochastic signals
        if (indicators.stoch_k < 20.0f) bullish_signals += 1.0f;
        else if (indicators.stoch_k > 80.0f) bearish_signals += 1.0f;
        signal_count++;
        
        // Williams %R signals
        if (indicators.williams_r < -80.0f) bullish_signals += 1.0f;
        else if (indicators.williams_r > -20.0f) bearish_signals += 1.0f;
        signal_count++;
        
        technical_score = (bullish_signals - bearish_signals) / signal_count;
        technical_score = (technical_score + 1.0f) / 2.0f;  // Normalize to [0,1]
    }
    
    void calculateRiskScore(const TechnicalAnalysis::TechnicalIndicators& indicators) {
        float risk_factors = 0.0f;
        int factor_count = 0;
        
        // Volatility risk
        if (indicators.volatility_20 > 0.5f) risk_factors += 1.0f;
        factor_count++;
        
        // ATR risk
        if (indicators.atr_14 > indicators.sma_20 * 0.05f) risk_factors += 1.0f;
        factor_count++;
        
        // Bollinger Band squeeze (low risk when squeezed)
        if (indicators.bb_squeeze < 0.02f) risk_factors -= 0.5f;
        factor_count++;
        
        risk_score = factor_count > 0 ? risk_factors / factor_count : 0.5f;
        risk_score = std::clamp(risk_score, 0.0f, 1.0f);
    }
    
    std::string generateBuyReasoning(const TechnicalAnalysis::TechnicalIndicators& indicators) {
        std::ostringstream reasoning;
        reasoning << "BUY signals: ";
        
        if (indicators.valid) {
            if (indicators.rsi_14 < 30.0f) reasoning << "RSI oversold ";
            if (indicators.macd > indicators.macd_signal) reasoning << "MACD bullish ";
            if (indicators.stoch_k < 20.0f) reasoning << "Stoch oversold ";
        }
        
        reasoning << "(conf: " << std::fixed << std::setprecision(2) << confidence << ")";
        return reasoning.str();
    }
    
    std::string generateSellReasoning(const TechnicalAnalysis::TechnicalIndicators& indicators) {
        std::ostringstream reasoning;
        reasoning << "SELL signals: ";
        
        if (indicators.valid) {
            if (indicators.rsi_14 > 70.0f) reasoning << "RSI overbought ";
            if (indicators.macd < indicators.macd_signal) reasoning << "MACD bearish ";
            if (indicators.stoch_k > 80.0f) reasoning << "Stoch overbought ";
        }
        
        reasoning << "(conf: " << std::fixed << std::setprecision(2) << confidence << ")";
        return reasoning.str();
    }
};

// Enhanced Main Trading Simulation with Comprehensive Analytics
int main(int argc, char* argv[]) {
    try {
        // Enhanced Configuration
        std::string data_dir = "highly_diverse_stock_data";
        int num_epochs = 5;
        bool detailed_logging = false;
        bool generate_charts = true;
        bool use_synthetic_data = true;
        
        if (argc > 1) data_dir = argv[1];
        if (argc > 2) num_epochs = std::stoi(argv[2]);
        if (argc > 3) detailed_logging = (std::string(argv[3]) == "verbose");
        if (argc > 4) generate_charts = (std::string(argv[4]) != "no-charts");
        if (argc > 5) use_synthetic_data = (std::string(argv[5]) != "no-synthetic");
        
        std::cout << "=== Advanced Neural Trading Simulation v2.0 ===" << std::endl;
        std::cout << "Data Directory: " << data_dir << std::endl;
        std::cout << "Epochs: " << num_epochs << std::endl;
        std::cout << "Detailed Logging: " << (detailed_logging ? "ON" : "OFF") << std::endl;
        std::cout << "Chart Generation: " << (generate_charts ? "ON" : "OFF") << std::endl;
        std::cout << "Synthetic Data: " << (use_synthetic_data ? "ON" : "OFF") << std::endl;
        std::cout << "=================================================" << std::endl;

        // CUDA Device Check
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess || device_count == 0) {
            std::cerr << "[ERROR] No CUDA-capable device detected. Exiting." << std::endl;
            return 1;
        }
        
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, 0);
        std::cout << "[CUDA] Using device: " << device_prop.name 
                  << " (Compute " << device_prop.major << "." << device_prop.minor << ")" << std::endl;
        std::cout << "[CUDA] Memory: " << device_prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        
        // Load available data files
        auto data_files = getAvailableDataFiles(data_dir);
        if (data_files.empty()) {
            std::cerr << "[FATAL] No CSV files found in " << data_dir << std::endl;
            return 1;
        }

        // Preload all market data once to avoid per-epoch disk I/O
        std::cout << "[INFO] Preloading market data..." << std::endl;
        auto all_data = preloadMarketData(data_files);
        
        // Calculate total data points
        size_t total_data_points = 0;
        for (const auto& data : all_data) {
            total_data_points += data.size();
        }
        std::cout << "[INFO] Loaded " << total_data_points << " total data points from " 
                  << data_files.size() << " files" << std::endl;

        // Initialize enhanced systems
        TradingPortfolio portfolio(1000000.0f);  // $1M starting capital
        AdvancedFeatureEngineer feature_engineer;
        ChartGenerator chart_generator;
        
        // Initialize metrics file
        metrics_file.open("trading_metrics.csv");
        metrics_file << "epoch,timestamp,symbol,action,price,portfolio_value,confidence,neural_score,"
                    << "technical_score,risk_score,rsi,macd,bb_position,volume_ratio,reward\n";
        
        // Initialize neural network (CUDA)
        std::cout << "[INIT] Initializing CUDA neural network..." << std::endl;
        initializeNetwork();
        
        // Random number generation for file shuffling
        std::random_device rd;
        std::mt19937 rng(rd());
        std::vector<size_t> file_indices(data_files.size());
        std::iota(file_indices.begin(), file_indices.end(), 0);
        
        // Performance tracking
        long long total_decisions = 0;
        long long profitable_decisions = 0;
        long long neural_buy_signals = 0;
        long long neural_sell_signals = 0;
        long long neural_hold_signals = 0;
        
        auto simulation_start = std::chrono::high_resolution_clock::now();
        
        // Main training loop with enhanced analytics
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n=== Epoch " << (epoch + 1) << "/" << num_epochs << " ===" << std::endl;
            
            auto epoch_start = std::chrono::high_resolution_clock::now();
            double epoch_start_value = portfolio.getTotalValue();
            
            // Shuffle files for better generalization
            std::shuffle(file_indices.begin(), file_indices.end(), rng);

            for (size_t idx : file_indices) {
                const auto& file_path = data_files[idx];
                const auto& market_data = all_data[idx];
                std::filesystem::path p(file_path);
                std::string symbol = p.stem().string();
                
                if (detailed_logging) {
                    std::cout << "[PROCESSING] " << symbol << " (" << market_data.size() << " points)" << std::endl;
                }
                
                if (market_data.empty()) continue;
                
                for (const auto& data_point : market_data) {
                    if (!data_point.valid) continue;
                    
                    // Engineer advanced features (80+ features)
                    auto features = feature_engineer.engineerFeatures(
                        data_point.open, data_point.high, data_point.low, 
                        data_point.close, data_point.volume
                    );
                    
                    // Compute reward signal
                    float reward = portfolio.computeReward();
                    dopamine_level = 0.99f * dopamine_level + 0.01f * reward;
                    
                    // Neural network forward pass
                    auto start_forward = std::chrono::high_resolution_clock::now();
                    auto raw_outputs = forwardCUDA(features, reward);
                    auto end_forward = std::chrono::high_resolution_clock::now();
                    
                    // Get current technical indicators for decision enhancement
                    TechnicalAnalysis::TechnicalIndicators indicators;
                    // Note: In a real implementation, we'd get these from feature_engineer
                    // For now, we'll simulate some basic indicators
                    if (features.size() >= 40) {
                        indicators.valid = true;
                        indicators.rsi_14 = (features[20] + 0.5f) * 100.0f;  // Denormalize RSI
                        indicators.macd = features[21];
                        indicators.macd_signal = features[22];
                        indicators.bb_upper = data_point.close * 1.02f;
                        indicators.bb_lower = data_point.close * 0.98f;
                        indicators.volatility_20 = std::abs(features[33]);
                        indicators.volume_ratio = std::exp(features[34]);
                        indicators.stoch_k = (features[27] + 0.5f) * 100.0f;
                        indicators.williams_r = features[29] * 100.0f - 50.0f;
                    }
                    
                    // Make enhanced trading decision with technical analysis
                    TradingDecision decision(raw_outputs, indicators);
                    
                    // Execute trade with enhanced portfolio management
                    float old_value = portfolio.getTotalValue();
                    bool trade_executed = portfolio.executeAction(
                        decision.action, data_point.close, decision.confidence, indicators
                    );
                    float new_value = portfolio.getTotalValue();
                    
                    // Update neural network with reward-based learning
                    auto start_learning = std::chrono::high_resolution_clock::now();
                    updateSynapticWeightsCUDA(reward);
                    auto end_learning = std::chrono::high_resolution_clock::now();
                    
                    // Track decision statistics
                    total_decisions++;
                    if (new_value > old_value) profitable_decisions++;
                    
                    if (decision.action == "buy") neural_buy_signals++;
                    else if (decision.action == "sell") neural_sell_signals++;
                    else neural_hold_signals++;
                    
                    // Record data for chart generation
                    if (generate_charts && total_decisions % 100 == 0) {
                        chart_generator.recordPortfolioValue(new_value);
                        chart_generator.recordNeuralActivity(
                            std::accumulate(raw_outputs.begin(), raw_outputs.end(), 0.0f) / raw_outputs.size()
                        );
                        chart_generator.recordRewardSignal(reward);
                        chart_generator.recordSynapticWeight(decision.confidence);
                        chart_generator.recordNeurotransmitter(dopamine_level);
                        chart_generator.recordLearningRate(0.001f);  // Placeholder
                    }
                    
                    // Write metrics to CSV
                    auto now = std::chrono::high_resolution_clock::now();
                    auto time_t = std::chrono::system_clock::to_time_t(
                        std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                            now - std::chrono::high_resolution_clock::now() + std::chrono::system_clock::now()));
                    
                    metrics_file << epoch << ","
                               << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                               << symbol << ","
                               << decision.action << ","
                               << std::fixed << std::setprecision(6) << data_point.close << ","
                               << new_value << ","
                               << decision.confidence << ","
                               << decision.neural_score << ","
                               << decision.technical_score << ","
                               << decision.risk_score << ",";
                    
                    if (indicators.valid) {
                        metrics_file << indicators.rsi_14 << ","
                                   << indicators.macd << ","
                                   << ((data_point.close - indicators.bb_lower) / 
                                       (indicators.bb_upper - indicators.bb_lower)) << ","
                                   << indicators.volume_ratio;
                    } else {
                        metrics_file << "0,0,0,1";
                    }
                    
                    metrics_file << "," << reward << "\n";
                    
                    // Detailed logging with enhanced information
                    if (detailed_logging && total_decisions % 1000 == 0) {
                        float forward_time = std::chrono::duration<float, std::milli>(
                            end_forward - start_forward).count();
                        float learning_time = std::chrono::duration<float, std::milli>(
                            end_learning - start_learning).count();
                        
                        std::cout << "[" << total_decisions << "] " << symbol 
                                  << " " << decision.action 
                                  << " @$" << std::fixed << std::setprecision(2) << data_point.close
                                  << " | Conf:" << decision.confidence
                                  << " Tech:" << decision.technical_score
                                  << " Risk:" << decision.risk_score
                                  << " | PnL:$" << (new_value - old_value)
                                  << " | Total:$" << new_value
                                  << " | Reward:" << reward
                                  << " | Time:" << forward_time << "ms/" << learning_time << "ms"
                                  << std::endl;
                        
                        if (!decision.reasoning.empty()) {
                            std::cout << "    Reasoning: " << decision.reasoning << std::endl;
                        }
                    }
                    
                    // Periodic performance summary
                    if (total_decisions % 10000 == 0) {
                        std::cout << "\n--- Progress Report (Decision " << total_decisions << ") ---" << std::endl;
                        std::cout << "Current Portfolio Value: $" << std::fixed << std::setprecision(2) 
                                  << portfolio.getTotalValue() << std::endl;
                        std::cout << "Return: " << portfolio.getReturnPercent() << "%" << std::endl;
                        std::cout << "Profitable Decisions: " << (100.0 * profitable_decisions / total_decisions) 
                                  << "%" << std::endl;
                        std::cout << "Signal Distribution - Buy: " << neural_buy_signals 
                                  << ", Sell: " << neural_sell_signals 
                                  << ", Hold: " << neural_hold_signals << std::endl;
                        std::cout << "Dopamine Level: " << dopamine_level << std::endl;
                        std::cout << "-----------------------------------------------\n" << std::endl;
                    }
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_duration = std::chrono::duration<double>(epoch_end - epoch_start).count();
            double epoch_return = (portfolio.getTotalValue() - epoch_start_value) / epoch_start_value * 100.0;
            
            std::cout << "\n=== Epoch " << (epoch + 1) << " Summary ===" << std::endl;
            std::cout << "Duration: " << std::setprecision(1) << epoch_duration << " seconds" << std::endl;
            std::cout << "Epoch Return: " << std::setprecision(2) << epoch_return << "%" << std::endl;
            std::cout << "Portfolio Value: $" << std::setprecision(2) << portfolio.getTotalValue() << std::endl;
            std::cout << "Sharpe Ratio: " << std::setprecision(3) << portfolio.getSharpeRatio() << std::endl;
            std::cout << "Max Drawdown: " << std::setprecision(2) << portfolio.getMaxDrawdown() * 100.0f << "%" << std::endl;
            std::cout << "=========================" << std::endl;
        }
        
        auto simulation_end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(simulation_end - simulation_start).count();
        
        // Final comprehensive results
        std::cout << "\n=== SIMULATION COMPLETE ===" << std::endl;
        std::cout << "Total Duration: " << std::fixed << std::setprecision(1) << total_duration << " seconds" << std::endl;
        std::cout << "Total Decisions: " << total_decisions << std::endl;
        std::cout << "Profitable Decisions: " << profitable_decisions
                  << " (" << std::fixed << std::setprecision(2)
                  << (100.0 * profitable_decisions / total_decisions) << "%)" << std::endl;
        std::cout << "Decisions per Second: " << std::fixed << std::setprecision(2)
                  << (total_decisions / total_duration) << std::endl;
        
        std::cout << "\n=== Signal Distribution ===" << std::endl;
        std::cout << "Buy Signals: " << neural_buy_signals 
                  << " (" << (100.0 * neural_buy_signals / total_decisions) << "%)" << std::endl;
        std::cout << "Sell Signals: " << neural_sell_signals 
                  << " (" << (100.0 * neural_sell_signals / total_decisions) << "%)" << std::endl;
        std::cout << "Hold Signals: " << neural_hold_signals 
                  << " (" << (100.0 * neural_hold_signals / total_decisions) << "%)" << std::endl;
        
        std::cout << "\n=== Neural Network State ===" << std::endl;
        std::cout << "Final Dopamine Level: " << std::fixed << std::setprecision(4) << dopamine_level << std::endl;
        
        // Enhanced portfolio summary
        portfolio.printSummary();
        
        // Generate comprehensive charts
        if (generate_charts) {
            std::cout << "\n=== Generating Charts ===" << std::endl;
            chart_generator.generateAllCharts("trading_charts");
        }
        
        // Close metrics file
        metrics_file.close();
        std::cout << "\nMetrics saved to trading_metrics.csv" << std::endl;
        
        // Final system cleanup
        std::cout << "\n=== Cleanup ===" << std::endl;
        cleanupNetwork();
        std::cout << "Neural network cleaned up successfully" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        if (metrics_file.is_open()) metrics_file.close();
        cleanupNetwork();
        return 1;
    }
}
