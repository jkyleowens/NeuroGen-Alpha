#include "TradingAgent.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <filesystem>

// Static constants initialization]
const float TradingAgent::DEFAULT_CONFIDENCE_THRESHOLD = 0.6f;
const float TradingAgent::REWARD_DECAY_FACTOR = 0.95f;

// =============================================================================
// FEATURE ENGINEER IMPLEMENTATION
// =============================================================================

FeatureEngineer::FeatureEngineer() {
    data_history_.reserve(MAX_HISTORY_SIZE);
}

std::vector<float> FeatureEngineer::engineerFeatures(const MarketData& data) {
    updateHistory(data);
    return engineerFeatures(data_history_);
}

std::vector<float> FeatureEngineer::engineerFeatures(const std::vector<MarketData>& history) {
    std::vector<float> features;
    
    if (history.empty()) {
        return std::vector<float>(15, 0.0f); // Return zero vector if no data
    }
    
    // Basic price features
    const MarketData& current = history.back();
    features.push_back(current.close);
    features.push_back(current.volume);
    features.push_back(current.getPriceChange());
    features.push_back(current.getPriceChangePercent());
    
    // Technical indicators (if we have enough history)
    if (history.size() >= 14) {
        std::vector<float> prices = getPrices();
        
        features.push_back(calculateRSI(prices, 14));
        features.push_back(calculateMACD(prices));
        features.push_back(calculateVolatility(prices, 20));
        features.push_back(calculateBollingerPosition(prices, 20));
        features.push_back(calculateMomentum(prices, 10));
        
        // Moving averages
        std::vector<float> ma5 = calculateMovingAverage(prices, 5);
        std::vector<float> ma10 = calculateMovingAverage(prices, 10);
        
        if (!ma5.empty() && !ma10.empty()) {
            features.push_back(ma5.back());
            features.push_back(ma10.back());
            features.push_back(ma5.back() - ma10.back()); // MA crossover signal
        } else {
            features.push_back(current.close);
            features.push_back(current.close);
            features.push_back(0.0f);
        }
        
        // Volume indicators
        features.push_back(calculateVolumeWeightedPrice(history, 10));
        features.push_back(calculateOnBalanceVolume(history));
        features.push_back(calculateStochastic(history, 14));
    } else {
        // Fill with basic values when insufficient history
        for (int i = 0; i < 11; ++i) {
            features.push_back(0.0f);
        }
    }
    
    return features;
}

float FeatureEngineer::calculateRSI(const std::vector<float>& prices, int period) {
    if (prices.size() < static_cast<size_t>(period + 1)) return 50.0f;
    
    float avgGain = 0.0f, avgLoss = 0.0f;
    
    // Calculate initial average gain/loss
    for (int i = 1; i <= period; ++i) {
        float change = prices[i] - prices[i-1];
        if (change > 0) avgGain += change;
        else avgLoss += (-change);
    }
    avgGain /= period;
    avgLoss /= period;
    
    if (avgLoss == 0.0f) return 100.0f;
    
    float rs = avgGain / avgLoss;
    return 100.0f - (100.0f / (1.0f + rs));
}

float FeatureEngineer::calculateMACD(const std::vector<float>& prices) {
    if (prices.size() < 26) return 0.0f;
    
    std::vector<float> ema12 = calculateMovingAverage(prices, 12);
    std::vector<float> ema26 = calculateMovingAverage(prices, 26);
    
    if (ema12.empty() || ema26.empty()) return 0.0f;
    
    return ema12.back() - ema26.back();
}

std::vector<float> FeatureEngineer::calculateMovingAverage(const std::vector<float>& prices, int period) {
    std::vector<float> ma;
    if (prices.size() < static_cast<size_t>(period)) return ma;
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < period; ++j) {
            sum += prices[i - j];
        }
        ma.push_back(sum / period);
    }
    
    return ma;
}

float FeatureEngineer::calculateVolatility(const std::vector<float>& prices, int period) const {
    if (prices.size() < static_cast<size_t>(period + 1)) return 0.0f;
    
    std::vector<float> returns;
    for (size_t i = prices.size() - period; i < prices.size() - 1; ++i) {
        if (prices[i] > 0) {
            returns.push_back((prices[i+1] - prices[i]) / prices[i]);
        }
    }
    
    if (returns.empty()) return 0.0f;
    
    float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
    float variance = 0.0f;
    
    for (float ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= returns.size();
    
    return std::sqrt(variance);
}

float FeatureEngineer::calculateBollingerPosition(const std::vector<float>& prices, int period) {
    if (prices.size() < static_cast<size_t>(period)) return 0.5f;
    
    std::vector<float> ma = calculateMovingAverage(prices, period);
    if (ma.empty()) return 0.5f;
    
    float volatility = calculateVolatility(prices, period);
    float current_price = prices.back();
    float middle = ma.back();
    float upper = middle + 2 * volatility;
    float lower = middle - 2 * volatility;
    
    if (upper == lower) return 0.5f;
    
    return (current_price - lower) / (upper - lower);
}

float FeatureEngineer::calculateVolumeWeightedPrice(const std::vector<MarketData>& data, int period) {
    if (data.size() < static_cast<size_t>(period)) return 0.0f;
    
    float totalVolume = 0.0f;
    float totalVolumePrice = 0.0f;
    
    for (size_t i = data.size() - period; i < data.size(); ++i) {
        float price = data[i].getTypicalPrice();
        totalVolumePrice += price * data[i].volume;
        totalVolume += data[i].volume;
    }
    
    return totalVolume > 0 ? totalVolumePrice / totalVolume : 0.0f;
}

float FeatureEngineer::calculateOnBalanceVolume(const std::vector<MarketData>& data) {
    if (data.size() < 2) return 0.0f;
    
    float obv = 0.0f;
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].close > data[i-1].close) {
            obv += data[i].volume;
        } else if (data[i].close < data[i-1].close) {
            obv -= data[i].volume;
        }
    }
    
    return obv;
}

float FeatureEngineer::calculateMomentum(const std::vector<float>& prices, int period) {
    if (prices.size() < static_cast<size_t>(period + 1)) return 0.0f;
    
    size_t current_idx = prices.size() - 1;
    size_t past_idx = current_idx - period;
    
    return prices[current_idx] - prices[past_idx];
}

float FeatureEngineer::calculateStochastic(const std::vector<MarketData>& data, int period) {
    if (data.size() < static_cast<size_t>(period)) return 50.0f;
    
    float highest = data[data.size() - period].high;
    float lowest = data[data.size() - period].low;
    
    for (size_t i = data.size() - period + 1; i < data.size(); ++i) {
        highest = std::max(highest, data[i].high);
        lowest = std::min(lowest, data[i].low);
    }
    
    float current = data.back().close;
    
    if (highest == lowest) return 50.0f;
    
    return ((current - lowest) / (highest - lowest)) * 100.0f;
}

float FeatureEngineer::getCurrentVolatility(int period) const {
    return calculateVolatility(getPrices(), period);
}

void FeatureEngineer::updateHistory(const MarketData& data) {
    data_history_.push_back(data);
    if (data_history_.size() > MAX_HISTORY_SIZE) {
        data_history_.erase(data_history_.begin());
    }
}

std::vector<float> FeatureEngineer::getPrices() const {
    std::vector<float> prices;
    prices.reserve(data_history_.size());
    for (const auto& data : data_history_) {
        prices.push_back(data.close);
    }
    return prices;
}

std::vector<float> FeatureEngineer::getVolumes() const {
    std::vector<float> volumes;
    volumes.reserve(data_history_.size());
    for (const auto& data : data_history_) {
        volumes.push_back(data.volume);
    }
    return volumes;
}

// =============================================================================
// PORTFOLIO MANAGER IMPLEMENTATION
// =============================================================================

PortfolioManager::PortfolioManager(float initial_capital)
    : initial_capital_(initial_capital)
    , available_capital_(initial_capital)
    , current_position_(0.0f)
    , position_entry_price_(0.0f)
    , realized_pnl_(0.0f)
    , peak_value_(initial_capital)
    , max_drawdown_(0.0f)
    , total_trades_(0)
    , winning_trades_(0)
    , max_position_size_(1.0f)
    , max_risk_per_trade_(0.02f) {
}

bool PortfolioManager::executeTrade(TradingAction action, float price, float position_size, float confidence) {
    if (price <= 0 || position_size < 0 || position_size > 1.0f) {
        return false;
    }
    
    bool trade_executed = false;
    
    switch (action) {
        case TradingAction::BUY: {
            if (current_position_ <= 0) { // Can buy when no position or short
                float trade_amount = available_capital_ * position_size;
                float shares = trade_amount / price;
                
                if (trade_amount <= available_capital_) {
                    if (current_position_ < 0) {
                        // Close short position first
                        realized_pnl_ += (-current_position_) * (position_entry_price_ - price);
                    }
                    
                    current_position_ += shares;
                    position_entry_price_ = price;
                    available_capital_ -= trade_amount;
                    trade_executed = true;
                }
            }
            break;
        }
        
        case TradingAction::SELL: {
            if (current_position_ >= 0) { // Can sell when long position or no position
                if (current_position_ > 0) {
                    // Close long position
                    float trade_value = current_position_ * price;
                    realized_pnl_ += current_position_ * (price - position_entry_price_);
                    available_capital_ += trade_value;
                    current_position_ = 0.0f;
                    trade_executed = true;
                } else {
                    // Enter short position
                    float trade_amount = available_capital_ * position_size;
                    float shares = trade_amount / price;
                    current_position_ = -shares;
                    position_entry_price_ = price;
                    trade_executed = true;
                }
            }
            break;
        }
        
        case TradingAction::HOLD:
        case TradingAction::NO_ACTION:
        default:
            // No trade executed
            break;
    }
    
    if (trade_executed) {
        total_trades_++;
        logTrade(action, price, position_size, confidence);
        updatePortfolioValue(price);
    }
    
    return trade_executed;
}

float PortfolioManager::calculateUnrealizedPnL(float current_price) const {
    if (current_position_ == 0.0f) return 0.0f;
    
    return current_position_ * (current_price - position_entry_price_);
}

float PortfolioManager::calculateRealizedPnL() const {
    return realized_pnl_;
}

float PortfolioManager::getTotalValue(float current_price) const {
    return available_capital_ + getPositionValue(current_price);
}

float PortfolioManager::getAvailableCapital() const {
    return available_capital_;
}

float PortfolioManager::getPositionValue(float current_price) const {
    return current_position_ * current_price;
}

float PortfolioManager::getSharpeRatio() const {
    if (trade_returns_.size() < 2) return 0.0f;
    
    float mean_return = std::accumulate(trade_returns_.begin(), trade_returns_.end(), 0.0f) / trade_returns_.size();
    
    float variance = 0.0f;
    for (float ret : trade_returns_) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= (trade_returns_.size() - 1);
    
    float std_dev = std::sqrt(variance);
    
    return std_dev > 0 ? mean_return / std_dev : 0.0f;
}

float PortfolioManager::getMaxDrawdown() const {
    return max_drawdown_;
}

float PortfolioManager::getTotalReturn() const {
    return initial_capital_ > 0 ? (available_capital_ + realized_pnl_ - initial_capital_) / initial_capital_ : 0.0f;
}

float PortfolioManager::getWinRate() const {
    return total_trades_ > 0 ? static_cast<float>(winning_trades_) / total_trades_ : 0.0f;
}

void PortfolioManager::printSummary(float current_price) const {
    float total_value = getTotalValue(current_price);
    float unrealized_pnl = calculateUnrealizedPnL(current_price);
    
    std::cout << "\n=== Portfolio Summary ===" << std::endl;
    std::cout << "Total Value: $" << std::fixed << std::setprecision(2) << total_value << std::endl;
    std::cout << "Available Capital: $" << available_capital_ << std::endl;
    std::cout << "Current Position: " << current_position_ << std::endl;
    std::cout << "Position Value: $" << getPositionValue(current_price) << std::endl;
    std::cout << "Realized P&L: $" << realized_pnl_ << std::endl;
    std::cout << "Unrealized P&L: $" << unrealized_pnl << std::endl;
    std::cout << "Total Return: " << (getTotalReturn() * 100) << "%" << std::endl;
    std::cout << "Total Trades: " << total_trades_ << std::endl;
    std::cout << "Win Rate: " << (getWinRate() * 100) << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << getSharpeRatio() << std::endl;
    std::cout << "Max Drawdown: " << (max_drawdown_ * 100) << "%" << std::endl;
}

void PortfolioManager::logTrade(TradingAction action, float price, float size, float confidence) {
    // Calculate trade return for the previous trade
    if (!portfolio_values_.empty()) {
        float previous_value = portfolio_values_.back();
        float current_value = getTotalValue(price);
        float trade_return = previous_value > 0 ? (current_value - previous_value) / previous_value : 0.0f;
        
        trade_returns_.push_back(trade_return);
        
        if (trade_return > 0) {
            winning_trades_++;
        }
    }
}

void PortfolioManager::updatePortfolioValue(float current_price) {
    float total_value = getTotalValue(current_price);
    portfolio_values_.push_back(total_value);
    
    // Update peak value and max drawdown
    if (total_value > peak_value_) {
        peak_value_ = total_value;
    } else {
        float drawdown = (peak_value_ - total_value) / peak_value_;
        max_drawdown_ = std::max(max_drawdown_, drawdown);
    }
}

// =============================================================================
// RISK MANAGER IMPLEMENTATION
// =============================================================================

RiskManager::RiskManager()
    : max_risk_per_trade_(0.02f)
    , max_position_size_(1.0f)
    , min_confidence_threshold_(0.6f)
    , max_volatility_threshold_(0.05f) {
}

bool RiskManager::isTradeAllowed(TradingAction action, float confidence, float volatility) {
    if (action == TradingAction::HOLD || action == TradingAction::NO_ACTION) {
        return true;
    }
    
    if (confidence < min_confidence_threshold_) {
        return false;
    }
    
    if (volatility > max_volatility_threshold_) {
        return false;
    }
    
    return true;
}

float RiskManager::calculatePositionSize(float confidence, float volatility, float available_capital) {
    // Kelly criterion inspired position sizing
    float base_size = max_position_size_ * confidence;
    
    // Adjust for volatility
    float volatility_adjustment = std::max(0.1f, 1.0f - volatility * 10.0f);
    
    float position_size = base_size * volatility_adjustment;
    
    return std::min(position_size, max_position_size_);
}

float RiskManager::calculateStopLoss(TradingAction action, float entry_price, float volatility) {
    float stop_distance = entry_price * std::max(volatility * 2.0f, 0.02f); // Minimum 2% stop loss
    
    if (action == TradingAction::BUY) {
        return entry_price - stop_distance;
    } else if (action == TradingAction::SELL) {
        return entry_price + stop_distance;
    }
    
    return entry_price;
}

float RiskManager::calculateTakeProfit(TradingAction action, float entry_price, float confidence) {
    float profit_distance = entry_price * confidence * 0.05f; // Up to 5% take profit based on confidence
    
    if (action == TradingAction::BUY) {
        return entry_price + profit_distance;
    } else if (action == TradingAction::SELL) {
        return entry_price - profit_distance;
    }
    
    return entry_price;
}

float RiskManager::calculateVaR(const std::vector<float>& returns, float confidence_level) {
    if (returns.empty()) return 0.0f;
    
    std::vector<float> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t index = static_cast<size_t>(confidence_level * sorted_returns.size());
    if (index >= sorted_returns.size()) index = sorted_returns.size() - 1;
    
    return -sorted_returns[index]; // VaR is typically reported as positive
}

float RiskManager::calculateMaxDrawdownRisk(float current_value, float peak_value) {
    if (peak_value <= 0) return 0.0f;
    
    return (peak_value - current_value) / peak_value;
}

// =============================================================================
// TRADING AGENT IMPLEMENTATION
// =============================================================================

TradingAgent::TradingAgent(const std::string& symbol)
    : symbol_(symbol)
    , is_trading_(false)
    , learning_enabled_(true)
    , update_interval_ms_(1000)
    , cumulative_reward_(0.0f) {
    
    initializeComponents();
}

TradingAgent::~TradingAgent() {
    stopTrading();
    if (trading_log_.is_open()) trading_log_.close();
    if (performance_log_.is_open()) performance_log_.close();
}

void TradingAgent::startTrading() {
    is_trading_ = true;
    std::cout << "[TradingAgent] Trading started for " << symbol_ << std::endl;
}

void TradingAgent::stopTrading() {
    is_trading_ = false;
    std::cout << "[TradingAgent] Trading stopped for " << symbol_ << std::endl;
}

bool TradingAgent::loadMarketData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[TradingAgent] Cannot open market data file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header if present
    std::getline(file, line);
    
    int loaded = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        MarketData data;
        
        try {
            // Parse CSV format: timestamp,open,high,low,close,volume
            std::getline(ss, data.datetime, ',');
            std::getline(ss, token, ','); data.open = std::stof(token);
            std::getline(ss, token, ','); data.high = std::stof(token);
            std::getline(ss, token, ','); data.low = std::stof(token);
            std::getline(ss, token, ','); data.close = std::stof(token);
            std::getline(ss, token, ','); data.volume = std::stof(token);
            
            if (data.validate()) {
                market_history_.push_back(data);
                loaded++;
            }
        } catch (const std::exception& e) {
            // Skip invalid lines
            continue;
        }
    }
    
    file.close();
    std::cout << "[TradingAgent] Loaded " << loaded << " market data points from " << filename << std::endl;
    return loaded > 0;
}

bool TradingAgent::loadMarketDataFromDirectory(const std::string& directory) {
    int total_loaded = 0;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() == ".csv") {
                if (loadMarketData(entry.path().string())) {
                    total_loaded++;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[TradingAgent] Error reading directory: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "[TradingAgent] Loaded data from " << total_loaded << " files" << std::endl;
    return total_loaded > 0;
}

void TradingAgent::processMarketData(const MarketData& data) {
    validateMarketData(data);
    
    market_history_.push_back(data);
    last_market_data_ = data;
    
    // Keep history manageable
    if (market_history_.size() > 1000) {
        market_history_.erase(market_history_.begin());
    }
    
    // Make trading decision if we have enough history
    if (market_history_.size() >= MIN_HISTORY_FOR_TRADING && is_trading_) {
        TradingDecision decision = makeDecision(data);
        
        // Calculate reward for previous decision
        if (!decision_history_.empty()) {
            float reward = calculateReward(decision_history_.back(), data);
            updateNeuralNetwork(decision_history_.back(), reward);
            logDecision(decision_history_.back(), reward);
        }
        
        decision_history_.push_back(decision);
    }
}

void TradingAgent::initializeNeuralNetwork() {
    // Initialize CUDA neural network interface
    // This would connect to the actual CUDA network implementation
    std::cout << "[TradingAgent] Neural network initialized" << std::endl;
}

TradingDecision TradingAgent::makeDecision(const MarketData& data) {
    TradingDecision decision;
    decision.timestamp = std::chrono::system_clock::now();
    
    // Prepare input features
    std::vector<float> neural_input = prepareNeuralInput(data);
    
    // Get neural network output (placeholder - would use actual CUDA network)
    std::vector<float> neural_outputs = {0.3f, 0.7f, 0.2f}; // [hold, buy, sell]
    
    // Interpret neural outputs
    decision.confidence = calculateConfidence(neural_outputs);
    decision.action = interpretNeuralOutput(neural_outputs, decision.confidence);
    
    // Risk management check
    if (risk_manager_->isTradeAllowed(decision.action, decision.confidence, 
                                     feature_engineer_->getCurrentVolatility())) {
        decision.position_size = risk_manager_->calculatePositionSize(
            decision.confidence, 
            feature_engineer_->getCurrentVolatility(),
            portfolio_manager_->getAvailableCapital()
        );
    } else {
        decision.action = TradingAction::HOLD;
        decision.position_size = 0.0f;
        decision.confidence = 0.0f;
    }
    
    decision.neural_outputs = neural_outputs;
    decision.rationale = "Neural network decision with confidence " + std::to_string(decision.confidence);
    
    // Execute trade if action is not HOLD
    if (decision.action != TradingAction::HOLD && decision.action != TradingAction::NO_ACTION) {
        portfolio_manager_->executeTrade(decision.action, data.close, decision.position_size, decision.confidence);
    }
    
    return decision;
}

void TradingAgent::updateNeuralNetwork(const TradingDecision& decision, float reward) {
    if (!learning_enabled_) return;
    
    reward_history_.push_back(reward);
    cumulative_reward_ += reward;
    
    // Send reward signal to neural network
    sendRewardToNetwork(reward);
    
    // Log neural network state
    logNeuralNetworkState();
}

float TradingAgent::calculateReward(const TradingDecision& decision, const MarketData& current_data) {
    float reward = 0.0f;
    
    if (decision.action == TradingAction::BUY && current_data.getPriceChange() > 0) {
        reward = current_data.getPriceChangePercent() * decision.confidence;
    } else if (decision.action == TradingAction::SELL && current_data.getPriceChange() < 0) {
        reward = -current_data.getPriceChangePercent() * decision.confidence;
    } else if (decision.action == TradingAction::HOLD) {
        reward = 0.1f; // Small positive reward for appropriate holding
    } else {
        reward = -std::abs(current_data.getPriceChangePercent()) * 0.5f; // Penalty for wrong direction
    }
    
    // Apply confidence scaling
    reward *= decision.confidence;
    
    // Risk-adjusted reward
    float volatility = feature_engineer_->getCurrentVolatility();
    if (volatility > 0) {
        reward /= (1.0f + volatility * 10.0f);
    }
    
    return reward;
}

void TradingAgent::evaluatePerformance() {
    if (market_history_.empty()) return;
    
    std::cout << "\n=== Trading Performance Evaluation ===" << std::endl;
    
    float current_price = market_history_.back().close;
    portfolio_manager_->printSummary(current_price);
    
    std::cout << "\nNeural Network Performance:" << std::endl;
    std::cout << "Total Decisions: " << decision_history_.size() << std::endl;
    std::cout << "Cumulative Reward: " << cumulative_reward_ << std::endl;
    
    if (!reward_history_.empty()) {
        float avg_reward = std::accumulate(reward_history_.begin(), reward_history_.end(), 0.0f) / reward_history_.size();
        std::cout << "Average Reward: " << avg_reward << std::endl;
    }
}

void TradingAgent::setRiskLevel(float level) {
    level = std::max(0.0f, std::min(1.0f, level));
    
    risk_manager_->setMaxRiskPerTrade(0.01f + level * 0.04f); // 1% to 5% risk per trade
    risk_manager_->setMaxPositionSize(0.5f + level * 0.5f);   // 50% to 100% max position
    risk_manager_->setMinConfidence(0.8f - level * 0.3f);     // 80% to 50% min confidence
    
    std::cout << "[TradingAgent] Risk level set to " << (level * 100) << "%" << std::endl;
}

void TradingAgent::printStatus() const {
    if (market_history_.empty()) {
        std::cout << "[TradingAgent] No market data loaded" << std::endl;
        return;
    }
    
    const MarketData& current = market_history_.back();
    
    std::cout << "\n=== Trading Agent Status ===" << std::endl;
    std::cout << "Symbol: " << symbol_ << std::endl;
    std::cout << "Trading: " << (is_trading_ ? "ACTIVE" : "INACTIVE") << std::endl;
    std::cout << "Learning: " << (learning_enabled_ ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Current Price: $" << std::fixed << std::setprecision(2) << current.close << std::endl;
    std::cout << "Price Change: " << current.getPriceChangePercent() << "%" << std::endl;
    std::cout << "Volume: " << current.volume << std::endl;
    std::cout << "Market Data Points: " << market_history_.size() << std::endl;
    std::cout << "Total Decisions: " << decision_history_.size() << std::endl;
    std::cout << "Cumulative Reward: " << cumulative_reward_ << std::endl;
    
    if (portfolio_manager_) {
        std::cout << "Portfolio Value: $" << portfolio_manager_->getTotalValue(current.close) << std::endl;
        std::cout << "Total Return: " << (portfolio_manager_->getTotalReturn() * 100) << "%" << std::endl;
    }
}

void TradingAgent::exportTradingLog(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[TradingAgent] Cannot create trading log file: " << filename << std::endl;
        return;
    }
    
    file << "timestamp,action,confidence,position_size,rationale\n";
    
    for (const auto& decision : decision_history_) {
        auto time_t = std::chrono::system_clock::to_time_t(decision.timestamp);
        file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ",";
        
        switch (decision.action) {
            case TradingAction::BUY: file << "BUY"; break;
            case TradingAction::SELL: file << "SELL"; break;
            case TradingAction::HOLD: file << "HOLD"; break;
            default: file << "NO_ACTION"; break;
        }
        
        file << "," << decision.confidence
             << "," << decision.position_size
             << "," << decision.rationale << "\n";
    }
    
    file.close();
    std::cout << "[TradingAgent] Trading log exported to: " << filename << std::endl;
}

void TradingAgent::exportPerformanceReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[TradingAgent] Cannot create performance report file: " << filename << std::endl;
        return;
    }
    
    file << "=== Trading Agent Performance Report ===\n\n";
    file << "Symbol: " << symbol_ << "\n";
    file << "Total Market Data Points: " << market_history_.size() << "\n";
    file << "Total Trading Decisions: " << decision_history_.size() << "\n";
    file << "Cumulative Neural Reward: " << cumulative_reward_ << "\n\n";
    
    if (portfolio_manager_ && !market_history_.empty()) {
        float current_price = market_history_.back().close;
        file << "Portfolio Performance:\n";
        file << "  Total Value: $" << portfolio_manager_->getTotalValue(current_price) << "\n";
        file << "  Total Return: " << (portfolio_manager_->getTotalReturn() * 100) << "%\n";
        file << "  Total Trades: " << portfolio_manager_->getTotalTrades() << "\n";
        file << "  Win Rate: " << (portfolio_manager_->getWinRate() * 100) << "%\n";
        file << "  Sharpe Ratio: " << portfolio_manager_->getSharpeRatio() << "\n";
        file << "  Max Drawdown: " << (portfolio_manager_->getMaxDrawdown() * 100) << "%\n";
    }
    
    file.close();
    std::cout << "[TradingAgent] Performance report exported to: " << filename << std::endl;
}

TradingAgent::TradingStatistics TradingAgent::getStatistics() const {
    TradingStatistics stats;
    
    if (portfolio_manager_) {
        stats.total_return = portfolio_manager_->getTotalReturn();
        stats.sharpe_ratio = portfolio_manager_->getSharpeRatio();
        stats.max_drawdown = portfolio_manager_->getMaxDrawdown();
        stats.win_rate = portfolio_manager_->getWinRate();
        stats.total_trades = portfolio_manager_->getTotalTrades();
    } else {
        stats.total_return = 0.0f;
        stats.sharpe_ratio = 0.0f;
        stats.max_drawdown = 0.0f;
        stats.win_rate = 0.0f;
        stats.total_trades = 0;
    }
    
    stats.avg_trade_duration_hours = 24.0f; // Placeholder
    stats.profit_factor = 1.0f; // Placeholder
    
    // Neural network confidence history
    for (const auto& decision : decision_history_) {
        stats.neural_confidence_history.push_back(decision.confidence);
    }
    
    stats.reward_history = reward_history_;
    
    return stats;
}

// =============================================================================
// PRIVATE METHODS IMPLEMENTATION
// =============================================================================

void TradingAgent::initializeComponents() {
    feature_engineer_ = std::make_unique<FeatureEngineer>();
    portfolio_manager_ = std::make_unique<PortfolioManager>(100000.0f); // $100k starting capital
    risk_manager_ = std::make_unique<RiskManager>();
    
    // Initialize logging
    trading_log_.open("trading_agent_log.csv");
    performance_log_.open("trading_performance.csv");
    
    std::cout << "[TradingAgent] Components initialized successfully" << std::endl;
}

void TradingAgent::logDecision(const TradingDecision& decision, float reward) {
    if (trading_log_.is_open()) {
        auto time_t = std::chrono::system_clock::to_time_t(decision.timestamp);
        trading_log_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                    << static_cast<int>(decision.action) << ","
                    << decision.confidence << ","
                    << decision.position_size << ","
                    << reward << "\n";
        trading_log_.flush();
    }
}

void TradingAgent::updateRewardSignals() {
    // Update pending rewards queue
    // This would handle delayed reward signals in a more sophisticated implementation
}

TradingAction TradingAgent::interpretNeuralOutput(const std::vector<float>& outputs, float& confidence) {
    if (outputs.size() < 3) {
        confidence = 0.0f;
        return TradingAction::HOLD;
    }
    
    // Find the action with highest output
    size_t max_idx = 0;
    for (size_t i = 1; i < outputs.size(); ++i) {
        if (outputs[i] > outputs[max_idx]) {
            max_idx = i;
        }
    }
    
    confidence = outputs[max_idx];
    
    switch (max_idx) {
        case 0: return TradingAction::HOLD;
        case 1: return TradingAction::BUY;
        case 2: return TradingAction::SELL;
        default: return TradingAction::HOLD;
    }
}

float TradingAgent::calculateConfidence(const std::vector<float>& outputs) {
    if (outputs.empty()) return 0.0f;
    
    float max_output = *std::max_element(outputs.begin(), outputs.end());
    float sum_outputs = std::accumulate(outputs.begin(), outputs.end(), 0.0f);
    
    // Softmax-like confidence calculation
    return sum_outputs > 0 ? max_output / sum_outputs : 0.0f;
}

void TradingAgent::validateMarketData(const MarketData& data) {
    if (!data.validate()) {
        throw std::invalid_argument("Invalid market data provided");
    }
}

std::vector<float> TradingAgent::prepareNeuralInput(const MarketData& data) {
    return feature_engineer_->engineerFeatures(data);
}

void TradingAgent::sendRewardToNetwork(float reward) {
    // This would send the reward signal to the CUDA neural network
    // For now, just log it
    neural_outputs_history_.push_back(reward);
}

void TradingAgent::logNeuralNetworkState() {
    // Log current neural network state for debugging/analysis
    if (performance_log_.is_open()) {
        performance_log_ << cumulative_reward_ << "," 
                        << (reward_history_.empty() ? 0.0f : reward_history_.back()) << "\n";
        performance_log_.flush();
    }
}
