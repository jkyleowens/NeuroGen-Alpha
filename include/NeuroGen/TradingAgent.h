#pragma once
#ifndef TRADING_AGENT_H
#define TRADING_AGENT_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <queue>
#include <chrono>
#include <fstream>
#include <functional>
#include <random>
#include <NeuroGen/cuda/NetworkCUDA_Interface.h>

// Forward declarations
struct MarketData;
class FeatureEngineer;
class PortfolioManager;
class RiskManager;

/**
 * @brief Market data structure for price information
 */
struct MarketData {
    std::string datetime;
    float open;
    float high;
    float low;
    float close;
    float volume;
    
    bool validate() const {
        return open > 0 && high > 0 && low > 0 && close > 0 && volume >= 0 &&
               high >= low && high >= open && high >= close && low <= open && low <= close;
    }
    
    float getTypicalPrice() const {
        return (high + low + close) / 3.0f;
    }
    
    float getPriceChange() const {
        return close - open;
    }
    
    float getPriceChangePercent() const {
        return open > 0 ? ((close - open) / open) * 100.0f : 0.0f;
    }
};

/**
 * @brief Trading decision types
 */
enum class TradingAction {
    HOLD = 0,
    BUY = 1,
    SELL = 2,
    NO_ACTION = 3
};

/**
 * @brief Trading decision with confidence and rationale
 */
struct TradingDecision {
    TradingAction action;
    float confidence;
    float position_size;  // Percentage of available capital/position
    std::string rationale;
    std::vector<float> neural_outputs;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Feature engineering for market data
 */
class FeatureEngineer {
public:
    FeatureEngineer();
    
    // Convert market data to neural network input features
    std::vector<float> engineerFeatures(const MarketData& data);
    std::vector<float> engineerFeatures(const std::vector<MarketData>& history);
    
    // Technical indicators
    float calculateRSI(const std::vector<float>& prices, int period = 14);
    float calculateMACD(const std::vector<float>& prices);
    std::vector<float> calculateMovingAverage(const std::vector<float>& prices, int period);
    float calculateVolatility(const std::vector<float>& prices, int period = 20) const;
    float calculateBollingerPosition(const std::vector<float>& prices, int period = 20);
    
    // Volume indicators
    float calculateVolumeWeightedPrice(const std::vector<MarketData>& data, int period = 10);
    float calculateOnBalanceVolume(const std::vector<MarketData>& data);
    
    // Momentum indicators
    float calculateMomentum(const std::vector<float>& prices, int period = 10);
    float calculateStochastic(const std::vector<MarketData>& data, int period = 14);
    
    // Convenience methods using internal data
    float getCurrentVolatility(int period = 20) const;
    
private:
    std::vector<MarketData> data_history_;
    static const int MAX_HISTORY_SIZE = 200;
    
    void updateHistory(const MarketData& data);
    std::vector<float> getPrices() const;
    std::vector<float> getVolumes() const;
};

/**
 * @brief Portfolio and position management
 */
class PortfolioManager {
public:
    PortfolioManager(float initial_capital = 100000.0f);
    
    // Portfolio operations
    bool executeTrade(TradingAction action, float price, float position_size, float confidence);
    float calculateUnrealizedPnL(float current_price) const;
    float calculateRealizedPnL() const;
    float getTotalValue(float current_price) const;
    float getAvailableCapital() const;
    float getInitialCapital() const { return initial_capital_; } // Added getter

    // Position management
    float getCurrentPosition() const { return current_position_; }
    float getPositionValue(float current_price) const;
    bool hasPosition() const { return std::abs(current_position_) > 1e-6f; }
    
    // Performance metrics
    float getSharpeRatio() const;
    float getMaxDrawdown() const;
    float getTotalReturn(float current_price) const; // Modified signature
    int getTotalTrades() const { return total_trades_; }
    float getWinRate() const;
    const std::vector<float>& getTradeReturns() const { return trade_returns_; } // Added getter
    
    // Portfolio state
    void printSummary(float current_price) const;
    // Modified signature to match definition
    void logTrade(TradingAction original_action, float current_market_price, float shares_executed, float confidence, float pnl_from_this_trade, float entry_price_of_closed_portion, bool was_closing_trade);
    void resetPortfolioState(); // Reset to initial state
    bool validatePortfolioState(float current_price) const; // Validate portfolio sanity
    
private:
    float initial_capital_;
    float available_capital_;
    float current_position_;       // Positive = long, negative = short
    float position_entry_price_;
    float realized_pnl_;
    float peak_value_;
    float max_drawdown_;
    
    // Trade tracking
    int total_trades_;
    int winning_trades_;
    std::vector<float> trade_returns_;
    std::vector<float> portfolio_values_;
    
    // Risk limits
    float max_position_size_;
    float max_risk_per_trade_;
    
    void updatePortfolioValue(float current_price);
};

/**
 * @brief Risk management and position sizing
 */
class RiskManager {
public:
    RiskManager();
    
    // Risk assessment
    bool isTradeAllowed(TradingAction action, float confidence, float volatility);
    float calculatePositionSize(float confidence, float volatility, float available_capital);
    float calculateStopLoss(TradingAction action, float entry_price, float volatility);
    float calculateTakeProfit(TradingAction action, float entry_price, float confidence);
    
    // Risk metrics
    float calculateVaR(const std::vector<float>& returns, float confidence_level = 0.05f);
    float calculateMaxDrawdownRisk(float current_value, float peak_value);
    
    // Configuration
    void setMaxRiskPerTrade(float risk) { max_risk_per_trade_ = risk; }
    void setMaxPositionSize(float size) { max_position_size_ = size; }
    void setMinConfidence(float conf) { min_confidence_threshold_ = conf; }
    
private:
    float max_risk_per_trade_;      // Maximum risk per trade (% of capital)
    float max_position_size_;       // Maximum position size (% of capital)
    float min_confidence_threshold_; // Minimum confidence to execute trade
    float max_volatility_threshold_; // Maximum volatility to allow trading
};

/**
 * @brief Main trading agent class
 */
class TradingAgent {
public:
    TradingAgent(const std::string& symbol = "BTCUSD");
    ~TradingAgent();
    
    // Main trading loop
    void startTrading();
    void stopTrading();
    bool isTrading() const { return is_trading_; }
    
    // Data processing
    bool loadMarketData(const std::string& filename);
    bool loadMarketDataFromDirectory(const std::string& directory);
    void processMarketData(const MarketData& data);
    
    // Neural network interface
    void initializeNeuralNetwork();
    TradingDecision makeDecision(const MarketData& data);
    void updateNeuralNetwork(const TradingDecision& decision, float reward);
    
    // Neural network persistence
    void saveNeuralNetworkState() const;
    bool saveNeuralNetworkState(const std::string& filename) const;
    bool loadNeuralNetworkState();
    bool loadNeuralNetworkState(const std::string& filename);
    void resetRewardTracking(); // Reset reward calculation tracking
    
    // Configuration
    void setSymbol(const std::string& symbol) { symbol_ = symbol; }
    void setUpdateInterval(int milliseconds) { update_interval_ms_ = milliseconds; }
    void setLearningMode(bool enabled) { learning_enabled_ = enabled; }
    void setRiskLevel(float level); // 0.0 = conservative, 1.0 = aggressive
    
    // Monitoring and logging
    void printStatus() const;
    void exportTradingLog(const std::string& filename) const;
    void exportPerformanceReport(const std::string& filename) const;
    void evaluatePerformance();
    
    // Statistics
    struct TradingStatistics {
        float total_return;
        float sharpe_ratio;
        float max_drawdown;
        float win_rate;
        int total_trades;
        float avg_trade_duration_hours;
        float profit_factor;
        float portfolio_value;
        std::vector<float> neural_confidence_history;
        std::vector<float> reward_history;
    };
    
    TradingStatistics getStatistics() const;
    std::vector<MarketData> getMarketHistory() const;
    void setMarketHistory(const std::vector<MarketData>& history);

private:
    // Core components
    std::unique_ptr<FeatureEngineer> feature_engineer_;
    std::unique_ptr<PortfolioManager> portfolio_manager_;
    std::unique_ptr<RiskManager> risk_manager_;
    
    // Trading state
    std::string symbol_;
    bool is_trading_;
    bool learning_enabled_;
    int update_interval_ms_;
    
    // Market data
    std::vector<MarketData> market_history_;
    MarketData last_market_data_;
    
    // Decision tracking
    std::vector<TradingDecision> decision_history_;
    std::queue<std::pair<TradingDecision, float>> pending_rewards_;
    
    // Performance tracking
    std::vector<float> reward_history_;
    std::vector<float> neural_outputs_history_;
    float cumulative_reward_;
    
    // Logging
    std::ofstream trading_log_;
    std::ofstream performance_log_;

    float epsilon_;
    static constexpr float EPSILON_DECAY = 0.9995f;
    
    // Neural network persistence
    bool autosave_enabled_;
    int network_save_interval_;
    int decisions_since_save_;
    std::string network_state_file_;
    
    // Private methods
    void initializeComponents();
    void logDecision(const TradingDecision& decision, float reward);
    void updateRewardSignals();
    TradingAction interpretNeuralOutput(const std::vector<float>& outputs, float& confidence);
    float calculateConfidence(const std::vector<float>& outputs);
    void validateMarketData(const MarketData& data);
    float calculateReward(const TradingDecision& last_decision, const MarketData& current_data);
    
    // Neural network interaction helpers
    std::vector<float> prepareNeuralInput(const MarketData& data);
    void sendRewardToNetwork(float reward);
    void logNeuralNetworkState();
    
    // Constants
    static const int MIN_HISTORY_FOR_TRADING = 20;
    static const float DEFAULT_CONFIDENCE_THRESHOLD;
    static const float REWARD_DECAY_FACTOR;
};

#endif // TRADING_AGENT_H