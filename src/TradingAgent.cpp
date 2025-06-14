#include <NeuroGen/TradingAgent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <thread>
#include <random>

// Constants
const float TradingAgent::DEFAULT_CONFIDENCE_THRESHOLD = 0.6f;
const float TradingAgent::REWARD_DECAY_FACTOR = 0.95f;

// ===============================================================================
// FeatureEngineer Implementation
// ===============================================================================

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
        return std::vector<float>(64, 0.0f); // Return 64 zero features
    }
    
    const MarketData& current = history.back();
    std::vector<float> prices = getPrices();
    std::vector<float> volumes = getVolumes();
    
    // Basic price features (8 features)
    features.push_back(current.open);
    features.push_back(current.high);
    features.push_back(current.low);
    features.push_back(current.close);
    features.push_back(current.volume);
    features.push_back(current.getPriceChange());
    features.push_back(current.getPriceChangePercent());
    features.push_back(current.getTypicalPrice());
    
    // Moving averages (12 features)
    if (prices.size() >= 5) {
        auto ma5 = calculateMovingAverage(prices, 5);
        features.push_back(ma5.empty() ? current.close : ma5.back());
        features.push_back(current.close - (ma5.empty() ? current.close : ma5.back()));
    } else {
        features.push_back(current.close);
        features.push_back(0.0f);
    }
    
    if (prices.size() >= 10) {
        auto ma10 = calculateMovingAverage(prices, 10);
        features.push_back(ma10.empty() ? current.close : ma10.back());
        features.push_back(current.close - (ma10.empty() ? current.close : ma10.back()));
    } else {
        features.push_back(current.close);
        features.push_back(0.0f);
    }
    
    if (prices.size() >= 20) {
        auto ma20 = calculateMovingAverage(prices, 20);
        features.push_back(ma20.empty() ? current.close : ma20.back());
        features.push_back(current.close - (ma20.empty() ? current.close : ma20.back()));
    } else {
        features.push_back(current.close);
        features.push_back(0.0f);
    }
    
    if (prices.size() >= 50) {
        auto ma50 = calculateMovingAverage(prices, 50);
        features.push_back(ma50.empty() ? current.close : ma50.back());
        features.push_back(current.close - (ma50.empty() ? current.close : ma50.back()));
    } else {
        features.push_back(current.close);
        features.push_back(0.0f);
    }
    
    // Technical indicators (16 features)
    features.push_back(calculateRSI(prices));
    features.push_back(calculateMACD(prices));
    features.push_back(calculateVolatility(prices));
    features.push_back(calculateBollingerPosition(prices));
    features.push_back(calculateMomentum(prices, 5));
    features.push_back(calculateMomentum(prices, 10));
    features.push_back(calculateStochastic(history));
    features.push_back(calculateVolumeWeightedPrice(history));
    features.push_back(calculateOnBalanceVolume(history));
    
    // Price ratios and relationships (7 features)
    features.push_back(current.high / current.low);
    features.push_back((current.close - current.low) / (current.high - current.low));
    features.push_back(current.volume / (volumes.empty() ? 1.0f : 
        std::accumulate(volumes.begin(), volumes.end(), 0.0f) / volumes.size()));
    
    // Recent price momentum (multiple timeframes) (8 features)
    if (prices.size() >= 3) {
        features.push_back((prices.back() - prices[prices.size()-3]) / prices[prices.size()-3]);
    } else {
        features.push_back(0.0f);
    }
    
    if (prices.size() >= 5) {
        features.push_back((prices.back() - prices[prices.size()-5]) / prices[prices.size()-5]);
    } else {
        features.push_back(0.0f);
    }
    
    if (prices.size() >= 10) {
        features.push_back((prices.back() - prices[prices.size()-10]) / prices[prices.size()-10]);
    } else {
        features.push_back(0.0f);
    }
    
    // Volume analysis (5 features)
    if (volumes.size() >= 5) {
        float avg_vol = std::accumulate(volumes.end()-5, volumes.end(), 0.0f) / 5.0f;
        features.push_back(current.volume / avg_vol);
    } else {
        features.push_back(1.0f);
    }
    
    // Market microstructure (8 features)
    features.push_back((current.close - current.open) / (current.high - current.low + 1e-8f));
    features.push_back((current.high - std::max(current.open, current.close)) / (current.high - current.low + 1e-8f));
    features.push_back((std::min(current.open, current.close) - current.low) / (current.high - current.low + 1e-8f));
    
    // Pad or truncate to exactly 64 features
    features.resize(64, 0.0f);
    
    // Normalize features to reasonable ranges
    for (size_t i = 0; i < features.size(); ++i) {
        if (std::isnan(features[i]) || std::isinf(features[i])) {
            features[i] = 0.0f;
        }
        // Clamp extreme values
        features[i] = std::max(-10.0f, std::min(10.0f, features[i]));
    }
    
    return features;
}

float FeatureEngineer::calculateRSI(const std::vector<float>& prices, int period) {
    if (static_cast<int>(prices.size()) < period + 1) return 50.0f; // Neutral RSI
    
    float gains = 0.0f, losses = 0.0f;
    
    for (int i = prices.size() - period; i < static_cast<int>(prices.size()); ++i) {
        float change = prices[i] - prices[i-1];
        if (change > 0) gains += change;
        else losses -= change;
    }
    
    float avg_gain = gains / period;
    float avg_loss = losses / period;
    
    if (avg_loss == 0.0f) return 100.0f;
    
    float rs = avg_gain / avg_loss;
    return 100.0f - (100.0f / (1.0f + rs));
}

float FeatureEngineer::calculateMACD(const std::vector<float>& prices) {
    if (prices.size() < 26) return 0.0f;
    
    auto ema12 = calculateMovingAverage(prices, 12);
    auto ema26 = calculateMovingAverage(prices, 26);
    
    if (ema12.empty() || ema26.empty()) return 0.0f;
    
    return ema12.back() - ema26.back();
}

std::vector<float> FeatureEngineer::calculateMovingAverage(const std::vector<float>& prices, int period) {
    std::vector<float> ma;
    if (static_cast<int>(prices.size()) < period) return ma;
    
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
    if (static_cast<int>(prices.size()) < period) return 0.0f;
    
    std::vector<float> returns;
    for (size_t i = prices.size() - period + 1; i < prices.size(); ++i) {
        returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
    }
    
    float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
    float variance = 0.0f;
    
    for (float ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    
    return std::sqrt(variance / returns.size()) * std::sqrt(252); // Annualized
}

float FeatureEngineer::calculateBollingerPosition(const std::vector<float>& prices, int period) {
    if (static_cast<int>(prices.size()) < period) return 0.5f;
    
    auto ma = calculateMovingAverage(prices, period);
    if (ma.empty()) return 0.5f;
    
    float volatility = calculateVolatility(prices, period);
    float current_price = prices.back();
    float middle_band = ma.back();
    float upper_band = middle_band + 2 * volatility;
    float lower_band = middle_band - 2 * volatility;
    
    return (current_price - lower_band) / (upper_band - lower_band + 1e-8f);
}

float FeatureEngineer::calculateMomentum(const std::vector<float>& prices, int period) {
    if (static_cast<int>(prices.size()) < period + 1) return 0.0f;
    
    return (prices.back() - prices[prices.size() - period - 1]) / prices[prices.size() - period - 1];
}

float FeatureEngineer::calculateStochastic(const std::vector<MarketData>& data, int period) {
    if (static_cast<int>(data.size()) < period) return 50.0f;
    
    float current_close = data.back().close;
    float lowest_low = data.back().low;
    float highest_high = data.back().high;
    
    for (size_t i = data.size() - period; i < data.size(); ++i) {
        lowest_low = std::min(lowest_low, data[i].low);
        highest_high = std::max(highest_high, data[i].high);
    }
    
    return (current_close - lowest_low) / (highest_high - lowest_low + 1e-8f) * 100.0f;
}

float FeatureEngineer::calculateVolumeWeightedPrice(const std::vector<MarketData>& data, int period) {
    if (static_cast<int>(data.size()) < period) return data.empty() ? 0.0f : data.back().close;
    
    float total_volume = 0.0f;
    float weighted_price = 0.0f;
    
    for (size_t i = data.size() - period; i < data.size(); ++i) {
        float typical_price = data[i].getTypicalPrice();
        weighted_price += typical_price * data[i].volume;
        total_volume += data[i].volume;
    }
    
    return total_volume > 0 ? weighted_price / total_volume : data.back().close;
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

void FeatureEngineer::updateHistory(const MarketData& data) {
    data_history_.push_back(data);
    if (data_history_.size() > MAX_HISTORY_SIZE) {
        data_history_.erase(data_history_.begin());
    }
}

std::vector<float> FeatureEngineer::getPrices() const {
    std::vector<float> prices;
    for (const auto& data : data_history_) {
        prices.push_back(data.close);
    }
    return prices;
}

std::vector<float> FeatureEngineer::getVolumes() const {
    std::vector<float> volumes;
    for (const auto& data : data_history_) {
        volumes.push_back(data.volume);
    }
    return volumes;
}

float FeatureEngineer::getCurrentVolatility(int period) const {
    std::vector<float> prices = getPrices();
    return calculateVolatility(prices, period);
}

// ===============================================================================
// PortfolioManager Implementation
// ===============================================================================

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
    , max_position_size_(0.5f)  // 50% max position
    , max_risk_per_trade_(0.02f) // 2% max risk per trade
{
    portfolio_values_.push_back(initial_capital);
}

bool PortfolioManager::executeTrade(TradingAction action, float price, float position_size, float confidence) {
    if (action == TradingAction::HOLD || action == TradingAction::NO_ACTION) {
        return true;
    }
    
    float trade_amount = available_capital_ * position_size;
    
    if (action == TradingAction::BUY) {
        if (current_position_ >= 0) {
            // Long position or neutral - increase long
            float shares_to_buy = trade_amount / price;
            
            if (shares_to_buy * price <= available_capital_) {
                float new_position = current_position_ + shares_to_buy;
                position_entry_price_ = (current_position_ * position_entry_price_ + shares_to_buy * price) / new_position;
                current_position_ = new_position;
                available_capital_ -= shares_to_buy * price;
                
                logTrade(action, price, shares_to_buy, confidence);
                return true;
            }
        } else {
            // Short position - reduce short or go long
            float shares_to_cover = std::min(-current_position_, trade_amount / price);
            float pnl = shares_to_cover * (position_entry_price_ - price);
            
            current_position_ += shares_to_cover;
            available_capital_ += shares_to_cover * position_entry_price_ + pnl;
            realized_pnl_ += pnl;
            
            if (shares_to_cover > 0) {
                logTrade(action, price, shares_to_cover, confidence);
                if (pnl > 0) winning_trades_++;
                trade_returns_.push_back(pnl / (shares_to_cover * position_entry_price_));
            }
            
            return true;
        }
    } else if (action == TradingAction::SELL) {
        if (current_position_ <= 0) {
            // Short position or neutral - increase short
            float shares_to_short = trade_amount / price;
            
            if (shares_to_short * price <= available_capital_) {
                float new_position = current_position_ - shares_to_short;
                position_entry_price_ = (-current_position_ * position_entry_price_ + shares_to_short * price) / (-new_position);
                current_position_ = new_position;
                available_capital_ += shares_to_short * price;
                
                logTrade(action, price, shares_to_short, confidence);
                return true;
            }
        } else {
            // Long position - reduce long or go short
            float shares_to_sell = std::min(current_position_, trade_amount / price);
            float pnl = shares_to_sell * (price - position_entry_price_);
            
            current_position_ -= shares_to_sell;
            available_capital_ += shares_to_sell * price;
            realized_pnl_ += pnl;
            
            if (shares_to_sell > 0) {
                logTrade(action, price, shares_to_sell, confidence);
                if (pnl > 0) winning_trades_++;
                trade_returns_.push_back(pnl / (shares_to_sell * position_entry_price_));
            }
            
            return true;
        }
    }
    
    return false;
}

float PortfolioManager::calculateUnrealizedPnL(float current_price) const {
    if (std::abs(current_position_) < 1e-6f) return 0.0f;
    
    return current_position_ * (current_price - position_entry_price_);
}

float PortfolioManager::getTotalValue(float current_price) const {
    float position_value = current_position_ * current_price;
    return available_capital_ + position_value;
}

float PortfolioManager::getSharpeRatio() const {
    if (trade_returns_.empty()) return 0.0f;
    
    float mean_return = std::accumulate(trade_returns_.begin(), trade_returns_.end(), 0.0f) / trade_returns_.size();
    
    float variance = 0.0f;
    for (float ret : trade_returns_) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    float std_dev = std::sqrt(variance / trade_returns_.size());
    
    return std_dev > 0 ? mean_return / std_dev : 0.0f;
}

float PortfolioManager::getMaxDrawdown() const {
    return max_drawdown_;
}

float PortfolioManager::getTotalReturn() const {
    return (getTotalValue(position_entry_price_) - initial_capital_) / initial_capital_;
}

float PortfolioManager::getWinRate() const {
    return total_trades_ > 0 ? static_cast<float>(winning_trades_) / total_trades_ : 0.0f;
}

float PortfolioManager::getAvailableCapital() const {
    return available_capital_;
}

void PortfolioManager::printSummary(float current_price) const {
    float total_value = getTotalValue(current_price);
    float unrealized_pnl = calculateUnrealizedPnL(current_price);
    
    std::cout << "\n=== Portfolio Summary ===" << std::endl;
    std::cout << "Initial Capital: $" << std::fixed << std::setprecision(2) << initial_capital_ << std::endl;
    std::cout << "Available Capital: $" << available_capital_ << std::endl;
    std::cout << "Current Position: " << current_position_ << " shares" << std::endl;
    std::cout << "Position Entry Price: $" << position_entry_price_ << std::endl;
    std::cout << "Current Price: $" << current_price << std::endl;
    std::cout << "Unrealized PnL: $" << unrealized_pnl << std::endl;
    std::cout << "Realized PnL: $" << realized_pnl_ << std::endl;
    std::cout << "Total Value: $" << total_value << std::endl;
    std::cout << "Total Return: " << (getTotalReturn() * 100.0f) << "%" << std::endl;
    std::cout << "Total Trades: " << total_trades_ << std::endl;
    std::cout << "Win Rate: " << (getWinRate() * 100.0f) << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << getSharpeRatio() << std::endl;
    std::cout << "Max Drawdown: " << (max_drawdown_ * 100.0f) << "%" << std::endl;
    std::cout << "========================" << std::endl;
}

void PortfolioManager::logTrade(TradingAction action, float price, float size, float confidence) {
    total_trades_++;
    
    std::string action_str;
    switch (action) {
        case TradingAction::BUY: action_str = "BUY"; break;
        case TradingAction::SELL: action_str = "SELL"; break;
        default: action_str = "UNKNOWN"; break;
    }
    
    std::cout << "[TRADE] " << action_str << " " << size << " @ $" << price 
              << " (Confidence: " << confidence << ")" << std::endl;
}

void PortfolioManager::updatePortfolioValue(float current_price) {
    float current_value = getTotalValue(current_price);
    portfolio_values_.push_back(current_value);
    
    if (current_value > peak_value_) {
        peak_value_ = current_value;
    } else {
        float drawdown = (peak_value_ - current_value) / peak_value_;
        max_drawdown_ = std::max(max_drawdown_, drawdown);
    }
}

// ===============================================================================
// RiskManager Implementation  
// ===============================================================================

RiskManager::RiskManager()
    : max_risk_per_trade_(0.02f)
    , max_position_size_(0.5f)
    , min_confidence_threshold_(0.6f)
    , max_volatility_threshold_(0.5f)
{
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
    (void)available_capital;  // Mark as intentionally unused for now
    
    float base_size = max_position_size_ * confidence;
    
    // Adjust for volatility
    float volatility_adjustment = 1.0f / (1.0f + volatility * 2.0f);
    
    float position_size = base_size * volatility_adjustment;
    
    return std::min(position_size, max_position_size_);
}

// ===============================================================================
// TradingAgent Implementation
// ===============================================================================

TradingAgent::TradingAgent(const std::string& symbol)
    : symbol_(symbol)
    , is_trading_(false)
    , learning_enabled_(true)
    , update_interval_ms_(100)
    , cumulative_reward_(0.0f)
{
    initializeComponents();
}

TradingAgent::~TradingAgent() {
    stopTrading();
    if (trading_log_.is_open()) trading_log_.close();
    if (performance_log_.is_open()) performance_log_.close();
}

void TradingAgent::initializeComponents() {
    feature_engineer_ = std::make_unique<FeatureEngineer>();
    portfolio_manager_ = std::make_unique<PortfolioManager>(100000.0f);
    risk_manager_ = std::make_unique<RiskManager>();
    
    // Open log files
    trading_log_.open("trading_agent_log.csv");
    trading_log_ << "timestamp,symbol,action,price,size,confidence,portfolio_value,reward\n";
    
    performance_log_.open("trading_performance.csv");
    performance_log_ << "timestamp,portfolio_value,unrealized_pnl,realized_pnl,total_trades,win_rate\n";
}

void TradingAgent::initializeNeuralNetwork() {
    std::cout << "[TradingAgent] Initializing neural network..." << std::endl;
    ::initializeNetwork(); // Call the global function from the interface
    std::cout << "[TradingAgent] Neural network initialized successfully." << std::endl;
}

bool TradingAgent::loadMarketDataFromDirectory(const std::string& directory) {
    std::cout << "[TradingAgent] Loading market data from: " << directory << std::endl;
    
    if (!std::filesystem::exists(directory)) {
        std::cerr << "[ERROR] Directory does not exist: " << directory << std::endl;
        return false;
    }
    
    std::vector<std::string> csv_files;
    
    // Find CSV files containing the symbol
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(symbol_) != std::string::npos && filename.find(".csv") != std::string::npos) {
                csv_files.push_back(entry.path().string());
            }
        }
    }
    
    if (csv_files.empty()) {
        // Fallback: load any CSV file
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csv_files.push_back(entry.path().string());
                break;
            }
        }
    }
    
    if (csv_files.empty()) {
        std::cerr << "[ERROR] No CSV files found in directory." << std::endl;
        return false;
    }
    
    // Sort files to ensure consistent loading order
    std::sort(csv_files.begin(), csv_files.end());
    
    for (const std::string& file : csv_files) {
        if (!loadMarketData(file)) {
            std::cerr << "[WARNING] Failed to load: " << file << std::endl;
        }
    }
    
    std::cout << "[TradingAgent] Loaded " << market_history_.size() << " data points from " 
              << csv_files.size() << " files." << std::endl;
    
    return !market_history_.empty();
}

bool TradingAgent::loadMarketData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line, header;
    std::getline(file, header); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> values;
        
        while (std::getline(ss, token, ',')) {
            values.push_back(token);
        }
        
        if (values.size() >= 6) {
            MarketData data;
            data.datetime = values[0];
            data.open = std::stof(values[1]);
            data.high = std::stof(values[2]);
            data.low = std::stof(values[3]);
            data.close = std::stof(values[4]);
            data.volume = std::stof(values[5]);
            
            if (data.validate()) {
                market_history_.push_back(data);
            }
        }
    }
    
    return true;
}

void TradingAgent::startTrading() {
    if (market_history_.size() < MIN_HISTORY_FOR_TRADING) {
        std::cerr << "[ERROR] Insufficient market data for trading. Need at least " 
                  << MIN_HISTORY_FOR_TRADING << " data points." << std::endl;
        return;
    }
    
    std::cout << "[TradingAgent] Starting autonomous trading..." << std::endl;
    is_trading_ = true;
    
    // Process historical data for training/backtesting
    for (size_t i = MIN_HISTORY_FOR_TRADING; i < market_history_.size() && is_trading_; ++i) {
        processMarketData(market_history_[i]);
        
        // Simulate real-time delay
        std::this_thread::sleep_for(std::chrono::milliseconds(update_interval_ms_));
    }
    
    std::cout << "[TradingAgent] Trading session completed." << std::endl;
    evaluatePerformance();
}

void TradingAgent::processMarketData(const MarketData& data) {
    validateMarketData(data);
    last_market_data_ = data;
    
    // Make trading decision
    TradingDecision decision = makeDecision(data);
    
    // Execute trade if allowed
    if (risk_manager_->isTradeAllowed(decision.action, decision.confidence, 
                                     feature_engineer_->getCurrentVolatility())) {
        
        float position_size = risk_manager_->calculatePositionSize(
            decision.confidence, 
            feature_engineer_->getCurrentVolatility(),
            portfolio_manager_->getAvailableCapital()
        );
        
        decision.position_size = position_size;
        
        portfolio_manager_->executeTrade(decision.action, data.close, position_size, decision.confidence);
    }
    
    // Calculate reward and update neural network
    float reward = calculateReward(decision, data);
    
    if (learning_enabled_) {
        updateNeuralNetwork(decision, reward);
    }
    
    // Log decision
    logDecision(decision, reward);
    decision_history_.push_back(decision);
    
    // Update statistics
    reward_history_.push_back(reward);
    cumulative_reward_ += reward;
}

TradingDecision TradingAgent::makeDecision(const MarketData& data) {
    TradingDecision decision;
    decision.timestamp = std::chrono::system_clock::now();
    
    // Prepare neural network input
    std::vector<float> features = prepareNeuralInput(data);
    
    // Get neural network output
    float current_reward = reward_history_.empty() ? 0.0f : reward_history_.back();
    decision.neural_outputs = ::forwardCUDA(features, current_reward);
    
    // Interpret neural network output
    decision.confidence = calculateConfidence(decision.neural_outputs);
    decision.action = interpretNeuralOutput(decision.neural_outputs, decision.confidence);
    
    // Generate rationale
    decision.rationale = "Neural network decision based on " + std::to_string(features.size()) + " features";
    
    return decision;
}

void TradingAgent::updateNeuralNetwork(const TradingDecision& decision, float reward) {
    // Log decision details for monitoring
    (void)decision;  // Mark as intentionally unused for now
    
    sendRewardToNetwork(reward);
    ::updateSynapticWeightsCUDA(reward);
}

float TradingAgent::calculateReward(const TradingDecision& decision, const MarketData& current_data) {
    float reward = 0.0f;
    
    // Base reward from portfolio performance
    static float last_portfolio_value = portfolio_manager_->getTotalValue(current_data.close);
    float current_portfolio_value = portfolio_manager_->getTotalValue(current_data.close);
    float portfolio_change = (current_portfolio_value - last_portfolio_value) / last_portfolio_value;
    
    // Reward based on portfolio improvement
    reward += portfolio_change * 100.0f; // Scale up
    
    // Penalty for high risk decisions with low confidence
    if (decision.confidence < 0.5f && decision.action != TradingAction::HOLD) {
        reward -= 0.1f;
    }
    
    // Bonus for high confidence correct decisions
    if (decision.confidence > 0.8f) {
        float price_change = 0.0f;
        if (market_history_.size() > 1) {
            price_change = (current_data.close - market_history_[market_history_.size()-2].close) / 
                          market_history_[market_history_.size()-2].close;
        }
        
        if ((decision.action == TradingAction::BUY && price_change > 0) ||
            (decision.action == TradingAction::SELL && price_change < 0)) {
            reward += 0.2f;
        }
    }
    
    // Update for next iteration
    last_portfolio_value = current_portfolio_value;
    
    // Clamp reward to reasonable range
    return std::max(-1.0f, std::min(1.0f, reward));
}

TradingAction TradingAgent::interpretNeuralOutput(const std::vector<float>& outputs, float& confidence) {
    if (outputs.empty()) {
        confidence = 0.0f;
        return TradingAction::HOLD;
    }
    
    // Find the action with highest activation
    auto max_it = std::max_element(outputs.begin(), outputs.end());
    int max_index = std::distance(outputs.begin(), max_it);
    
    confidence = *max_it;
    
    // Map neural output indices to actions
    switch (max_index % 4) {
        case 0: return TradingAction::HOLD;
        case 1: return TradingAction::BUY;
        case 2: return TradingAction::SELL;
        case 3: return TradingAction::NO_ACTION;
        default: return TradingAction::HOLD;
    }
}

float TradingAgent::calculateConfidence(const std::vector<float>& outputs) {
    if (outputs.empty()) return 0.0f;
    
    // Use softmax-like confidence calculation
    float max_val = *std::max_element(outputs.begin(), outputs.end());
    float sum_exp = 0.0f;
    
    for (float val : outputs) {
        sum_exp += std::exp(val - max_val);
    }
    
    return std::exp(max_val - max_val) / sum_exp; // This is just exp(0)/sum_exp = 1/sum_exp of the max
}

std::vector<float> TradingAgent::prepareNeuralInput(const MarketData& data) {
    return feature_engineer_->engineerFeatures(data);
}

void TradingAgent::sendRewardToNetwork(float reward) {
    // The reward is sent through the updateSynapticWeightsCUDA call
    // This is just for logging/monitoring
    (void)reward;  // Mark as intentionally unused for now
}

void TradingAgent::logDecision(const TradingDecision& decision, float reward) {
    if (trading_log_.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::string action_str;
        switch (decision.action) {
            case TradingAction::HOLD: action_str = "HOLD"; break;
            case TradingAction::BUY: action_str = "BUY"; break;
            case TradingAction::SELL: action_str = "SELL"; break;
            case TradingAction::NO_ACTION: action_str = "NO_ACTION"; break;
        }
        
        trading_log_ << time_t << "," << symbol_ << "," << action_str << "," 
                     << last_market_data_.close << "," << decision.position_size << ","
                     << decision.confidence << "," 
                     << portfolio_manager_->getTotalValue(last_market_data_.close) << ","
                     << reward << "\n";
        trading_log_.flush();
    }
}

void TradingAgent::evaluatePerformance() {
    std::cout << "\n=== Trading Agent Performance Report ===" << std::endl;
    portfolio_manager_->printSummary(last_market_data_.close);
    
    std::cout << "\nReward Statistics:" << std::endl;
    std::cout << "Total Decisions: " << decision_history_.size() << std::endl;
    std::cout << "Cumulative Reward: " << cumulative_reward_ << std::endl;
    
    if (!reward_history_.empty()) {
        float avg_reward = std::accumulate(reward_history_.begin(), reward_history_.end(), 0.0f) / reward_history_.size();
        std::cout << "Average Reward: " << avg_reward << std::endl;
    }
    
    std::cout << "=======================================" << std::endl;
}

void TradingAgent::stopTrading() {
    is_trading_ = false;
}

void TradingAgent::exportTradingLog(const std::string& filename) const {
    std::cout << "[TradingAgent] Exporting trading log to: " << filename << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot create trading log file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "timestamp,symbol,action,price,position_size,confidence,portfolio_value,reward\n";
    
    // Write decision data
    for (size_t i = 0; i < decision_history_.size(); ++i) {
        const auto& decision = decision_history_[i];
        float reward = (i < reward_history_.size()) ? reward_history_[i] : 0.0f;
        
        auto time_t = std::chrono::system_clock::to_time_t(decision.timestamp);
        
        std::string action_str;
        switch (decision.action) {
            case TradingAction::HOLD: action_str = "HOLD"; break;
            case TradingAction::BUY: action_str = "BUY"; break;
            case TradingAction::SELL: action_str = "SELL"; break;
            case TradingAction::NO_ACTION: action_str = "NO_ACTION"; break;
        }
        
        file << time_t << "," << symbol_ << "," << action_str << ","
             << last_market_data_.close << "," << decision.position_size << ","
             << decision.confidence << ","
             << portfolio_manager_->getTotalValue(last_market_data_.close) << ","
             << reward << "\n";
    }
    
    file.close();
    std::cout << "[TradingAgent] Trading log exported successfully." << std::endl;
}

void TradingAgent::exportPerformanceReport(const std::string& filename) const {
    std::cout << "[TradingAgent] Exporting performance report to: " << filename << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot create performance report file: " << filename << std::endl;
        return;
    }
    
    auto stats = getStatistics();
    
    file << "=== NeuroGen-Alpha Trading Agent Performance Report ===" << std::endl;
    file << "Generated: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
    file << "Symbol: " << symbol_ << std::endl;
    file << "========================================================" << std::endl;
    file << std::endl;
    
    file << "PORTFOLIO PERFORMANCE:" << std::endl;
    file << "Total Return: " << (stats.total_return * 100.0f) << "%" << std::endl;
    file << "Sharpe Ratio: " << stats.sharpe_ratio << std::endl;
    file << "Maximum Drawdown: " << (stats.max_drawdown * 100.0f) << "%" << std::endl;
    file << "Win Rate: " << (stats.win_rate * 100.0f) << "%" << std::endl;
    file << "Total Trades: " << stats.total_trades << std::endl;
    file << std::endl;
    
    file << "NEURAL NETWORK PERFORMANCE:" << std::endl;
    file << "Total Decisions: " << decision_history_.size() << std::endl;
    file << "Cumulative Reward: " << cumulative_reward_ << std::endl;
    
    if (!reward_history_.empty()) {
        float avg_reward = std::accumulate(reward_history_.begin(), reward_history_.end(), 0.0f) / reward_history_.size();
        file << "Average Reward: " << avg_reward << std::endl;
        
        float max_reward = *std::max_element(reward_history_.begin(), reward_history_.end());
        float min_reward = *std::min_element(reward_history_.begin(), reward_history_.end());
        file << "Max Reward: " << max_reward << std::endl;
        file << "Min Reward: " << min_reward << std::endl;
    }
    
    if (!stats.neural_confidence_history.empty()) {
        float avg_confidence = std::accumulate(stats.neural_confidence_history.begin(), 
                                              stats.neural_confidence_history.end(), 0.0f) / 
                              stats.neural_confidence_history.size();
        file << "Average Confidence: " << avg_confidence << std::endl;
    }
    
    file << std::endl;
    file << "TRADING BEHAVIOR ANALYSIS:" << std::endl;
    
    // Count actions
    int hold_count = 0, buy_count = 0, sell_count = 0, no_action_count = 0;
    for (const auto& decision : decision_history_) {
        switch (decision.action) {
            case TradingAction::HOLD: hold_count++; break;
            case TradingAction::BUY: buy_count++; break;
            case TradingAction::SELL: sell_count++; break;
            case TradingAction::NO_ACTION: no_action_count++; break;
        }
    }
    
    int total_decisions = decision_history_.size();
    if (total_decisions > 0) {
        file << "Hold Actions: " << hold_count << " (" << (100.0f * hold_count / total_decisions) << "%)" << std::endl;
        file << "Buy Actions: " << buy_count << " (" << (100.0f * buy_count / total_decisions) << "%)" << std::endl;
        file << "Sell Actions: " << sell_count << " (" << (100.0f * sell_count / total_decisions) << "%)" << std::endl;
        file << "No Action: " << no_action_count << " (" << (100.0f * no_action_count / total_decisions) << "%)" << std::endl;
    }
    
    file << std::endl;
    file << "=== End of Report ===" << std::endl;
    
    file.close();
    std::cout << "[TradingAgent] Performance report exported successfully." << std::endl;
}

void TradingAgent::validateMarketData(const MarketData& data) {
    if (!data.validate()) {
        throw std::runtime_error("Invalid market data received");
    }
}

void TradingAgent::printStatus() const {
    std::cout << "\n=== Trading Agent Status ===" << std::endl;
    std::cout << "Symbol: " << symbol_ << std::endl;
    std::cout << "Trading: " << (is_trading_ ? "Active" : "Inactive") << std::endl;
    std::cout << "Learning: " << (learning_enabled_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Market Data Points: " << market_history_.size() << std::endl;
    std::cout << "Decisions Made: " << decision_history_.size() << std::endl;
    std::cout << "Cumulative Reward: " << cumulative_reward_ << std::endl;
    if (!last_market_data_.datetime.empty()) {
        std::cout << "Last Price: $" << last_market_data_.close << std::endl;
        std::cout << "Portfolio Value: $" << portfolio_manager_->getTotalValue(last_market_data_.close) << std::endl;
    }
    std::cout << "============================" << std::endl;
}

TradingAgent::TradingStatistics TradingAgent::getStatistics() const {
    TradingStatistics stats;
    stats.total_return = portfolio_manager_->getTotalReturn();
    stats.sharpe_ratio = portfolio_manager_->getSharpeRatio();
    stats.max_drawdown = portfolio_manager_->getMaxDrawdown();
    stats.win_rate = portfolio_manager_->getWinRate();
    stats.total_trades = portfolio_manager_->getTotalTrades();
    stats.reward_history = reward_history_;
    
    // Extract confidence values from decision history
    for (const auto& decision : decision_history_) {
        stats.neural_confidence_history.push_back(decision.confidence);
    }
    
    return stats;
}
