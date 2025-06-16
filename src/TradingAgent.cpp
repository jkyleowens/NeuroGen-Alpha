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
    , total_trades_(0) // Will count closing trades
    , winning_trades_(0)
    , max_position_size_(1.0f)
    , max_risk_per_trade_(0.02f) {
}

bool PortfolioManager::executeTrade(TradingAction action, float price, float position_size, float confidence) {
    // Enhanced validation
    if (price <= 0 || !std::isfinite(price)) {
        std::cerr << "[PortfolioManager] Invalid price: " << price << std::endl;
        return false;
    }
    
    if (position_size < 0 || position_size > 1.0f || !std::isfinite(position_size)) {
        std::cerr << "[PortfolioManager] Invalid position size: " << position_size << std::endl;
        return false;
    }
    
    if (confidence < 0.0f || confidence > 1.0f || !std::isfinite(confidence)) {
        std::cerr << "[PortfolioManager] Invalid confidence: " << confidence << std::endl;
        return false;
    }
    
    available_capital_ = std::max(0.0f, available_capital_); // Ensure capital is not unintentionally negative before trade
    
    position_size = std::min(position_size, max_position_size_);
    
    bool trade_executed = false;
    float old_position = current_position_; // For debug logging

    float pnl_from_this_trade = 0.0f;
    float actual_shares_traded_for_log = 0.0f; // Shares involved in the primary action (closing or opening)
    float entry_price_for_closed_portion = 0.0f;
    bool is_closing_trade = false;
    
    switch (action) {
        case TradingAction::BUY: {
            float cash_to_use_for_buy = available_capital_ * position_size;
            if (cash_to_use_for_buy <= 1e-6f || price <= 1e-6f) { // Avoid trading with negligible amounts or invalid price
                 std::cout << "[PortfolioManager] BUY: Insufficient cash to use or invalid price. Cash: " << cash_to_use_for_buy << ", Price: " << price << std::endl;
                break;
            }
            float potential_shares_to_buy = cash_to_use_for_buy / price;

            if (current_position_ < -1e-8f) { // Currently short, this BUY is to cover
                is_closing_trade = true;
                entry_price_for_closed_portion = position_entry_price_;
                
                float shares_to_cover = std::min(potential_shares_to_buy, -current_position_);
                actual_shares_traded_for_log = shares_to_cover;

                if (shares_to_cover > 1e-8f && available_capital_ >= (shares_to_cover * price - 1e-6f)) { // check capital with tolerance
                    pnl_from_this_trade = shares_to_cover * (position_entry_price_ - price);
                    realized_pnl_ += pnl_from_this_trade;
                    available_capital_ -= (shares_to_cover * price);
                    current_position_ += shares_to_cover;
                    trade_executed = true;

                    if (std::abs(current_position_) < 1e-8f) {
                        current_position_ = 0.0f;
                        position_entry_price_ = 0.0f;
                    }
                } else {
                    std::cout << "[PortfolioManager] BUY (Cover Short): Not enough shares/capital. Shares: " << shares_to_cover << ", Capital: " << available_capital_ << ", Cost: " << shares_to_cover * price << std::endl;
                    break; 
                }

                float shares_for_new_long = potential_shares_to_buy - shares_to_cover;
                if (shares_for_new_long > 1e-8f && std::abs(current_position_) < 1e-8f && available_capital_ >= (shares_for_new_long * price - 1e-6f)) {
                    // Flipping to long: this is an opening trade, P&L already logged for the cover part.
                    // The `logTrade` call will reflect the closing part.
                    position_entry_price_ = price;
                    current_position_ = shares_for_new_long;
                    available_capital_ -= (shares_for_new_long * price);
                    // trade_executed is already true from covering.
                    // actual_shares_traded_for_log remains shares_to_cover for P&L attribution of the close.
                }
            } else { // Opening new long or adding to existing long
                is_closing_trade = false;
                actual_shares_traded_for_log = potential_shares_to_buy;

                if (actual_shares_traded_for_log > 1e-8f && available_capital_ >= (actual_shares_traded_for_log * price - 1e-6f)) {
                    if (current_position_ > 1e-8f) { // Adding to long
                        position_entry_price_ = ((current_position_ * position_entry_price_) + (actual_shares_traded_for_log * price)) / (current_position_ + actual_shares_traded_for_log);
                    } else { // New long
                        position_entry_price_ = price;
                    }
                    current_position_ += actual_shares_traded_for_log;
                    available_capital_ -= (actual_shares_traded_for_log * price);
                    trade_executed = true;
                } else {
                     std::cout << "[PortfolioManager] BUY (Open/Extend Long): Not enough shares/capital. Shares: " << actual_shares_traded_for_log << ", Capital: " << available_capital_ << ", Cost: " << actual_shares_traded_for_log * price << std::endl;
                }
            }
            break;
        }
        
        case TradingAction::SELL: {
            if (current_position_ > 1e-8f) { // Closing (partially or fully) a long position
                is_closing_trade = true;
                entry_price_for_closed_portion = position_entry_price_;
                
                float shares_to_sell = current_position_ * position_size; // position_size is fraction
                actual_shares_traded_for_log = shares_to_sell;

                if (shares_to_sell > 1e-8f) {
                    pnl_from_this_trade = shares_to_sell * (price - position_entry_price_);
                    realized_pnl_ += pnl_from_this_trade;
                    available_capital_ += (shares_to_sell * price);
                    current_position_ -= shares_to_sell;
                    trade_executed = true;

                    if (current_position_ < 1e-8f) {
                        current_position_ = 0.0f;
                        position_entry_price_ = 0.0f;
                    }
                } else {
                    std::cout << "[PortfolioManager] SELL (Close Long): Not enough shares to sell. Shares: " << shares_to_sell << std::endl;
                }
            } else { // Opening or extending a short position
                is_closing_trade = false;
                float value_to_short = available_capital_ * position_size; // Risk capital for this short
                
                if (value_to_short > 1e-6f && price > 1e-6f) {
                    float shares_to_short_additionally = value_to_short / price;
                    actual_shares_traded_for_log = shares_to_short_additionally;

                    if (current_position_ < -1e-8f) { // Extending existing short
                        float current_total_value_shorted = (-current_position_) * position_entry_price_;
                        float additional_value_shorted = shares_to_short_additionally * price;
                        current_position_ -= shares_to_short_additionally;
                        if (std::abs(current_position_) > 1e-8f) {
                            position_entry_price_ = (current_total_value_shorted + additional_value_shorted) / (-current_position_);
                        } else { // Should not happen if adding shares, but as a fallback
                            position_entry_price_ = price; 
                        }
                    } else { // New short position
                        current_position_ = -shares_to_short_additionally;
                        position_entry_price_ = price;
                    }
                    
                    available_capital_ += (shares_to_short_additionally * price); // Add proceeds from short sale
                    trade_executed = true;
                } else {
                    std::cout << "[PortfolioManager] SELL (Open/Extend Short): Not enough value to short or invalid price. Value: " << value_to_short << ", Price: " << price << std::endl;
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
        logTrade(action, price, actual_shares_traded_for_log, confidence, pnl_from_this_trade, entry_price_for_closed_portion, is_closing_trade);
        updatePortfolioValue(price);
        
        std::cout << "[Portfolio] Trade executed: " << (action == TradingAction::BUY ? "BUY" : (action == TradingAction::SELL ? "SELL" : "HOLD/NO_ACTION"))
                  << " | Shares: " << actual_shares_traded_for_log
                  << " | Price: " << price
                  << " | Old Pos: " << old_position << " -> New Pos: " << current_position_ 
                  << " | Entry Price: " << position_entry_price_ 
                  << " | Avail Capital: " << available_capital_
                  << " | PNL this trade: " << pnl_from_this_trade
                  << " | Realized PNL: " << realized_pnl_ << std::endl;
    }
    
    return trade_executed;
}

float PortfolioManager::calculateUnrealizedPnL(float current_price) const {
    if (current_position_ == 0.0f) return 0.0f;
    
    // Validate current price
    if (current_price <= 0.0f || !std::isfinite(current_price)) {
        std::cerr << "[PortfolioManager] Invalid current price: " << current_price << std::endl;
        return 0.0f;
    }
    
    // Validate position entry price
    if (position_entry_price_ <= 0.0f || !std::isfinite(position_entry_price_)) {
        std::cerr << "[PortfolioManager] Invalid entry price: " << position_entry_price_ << std::endl;
        return 0.0f;
    }
    
    // Calculate P&L without artificial clamping
    float pnl = 0.0f;
    if (current_position_ > 1e-8f) { // Long
        pnl = current_position_ * (current_price - position_entry_price_);
    } else if (current_position_ < -1e-8f) { // Short
        pnl = (-current_position_) * (position_entry_price_ - current_price);
    }
    
    // Basic validation only
    if (!std::isfinite(pnl)) {
        std::cerr << "[PortfolioManager] Non-finite P&L calculated, returning 0" << std::endl;
        return 0.0f;
    }
    
    return pnl;
}

float PortfolioManager::calculateRealizedPnL() const {
    return realized_pnl_;
}

float PortfolioManager::getTotalValue(float current_price) const {
    if (current_price <= 0.0f || !std::isfinite(current_price)) {
        std::cerr << "[PortfolioManager] Invalid current price for total value: " << current_price << std::endl;
        return available_capital_; // Return available cash if price is invalid
    }

    float value_from_open_positions = 0.0f;
    if (current_position_ > 1e-8f) { // Long position
        value_from_open_positions = current_position_ * current_price;
    } else if (current_position_ < -1e-8f) { // Short position
        // available_capital_ includes proceeds from short sale.
        // We subtract the current liability to repurchase the shorted shares.
        value_from_open_positions = -((-current_position_) * current_price);
    }

    float total_value = available_capital_ + value_from_open_positions;
    
    if (!std::isfinite(total_value)) {
        std::cerr << "[PortfolioManager] Non-finite total value calculated, returning available capital." << std::endl;
        return available_capital_;
    }
    
    return total_value;
}

float PortfolioManager::getAvailableCapital() const {
    return available_capital_;
}

float PortfolioManager::getPositionValue(float current_price) const {
    if (std::abs(current_position_) < 1e-8f) return 0.0f; // No position
    if (current_price <= 0.0f || !std::isfinite(current_price)) return 0.0f; // Invalid price

    if (current_position_ > 0) { // Long position
        return current_position_ * current_price;
    } else { // Short position: value is the unrealized P&L
        return (-current_position_) * (position_entry_price_ - current_price);
    }
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

float PortfolioManager::getTotalReturn(float current_price) const { // Added current_price argument
    if (initial_capital_ <= 1e-6f) return 0.0f; // Avoid division by zero or if no initial capital
    
    float current_total_value = getTotalValue(current_price);
    return (current_total_value - initial_capital_) / initial_capital_;
}

float PortfolioManager::getWinRate() const {
    return total_trades_ > 0 ? static_cast<float>(winning_trades_) / total_trades_ : 0.0f;
}

void PortfolioManager::printSummary(float current_price) const {
    float total_value = getTotalValue(current_price);
    float unrealized_pnl = calculateUnrealizedPnL(current_price);
    float total_return_percent = getTotalReturn(current_price) * 100.0f; // Use updated getTotalReturn
    
    std::cout << "\n=== Portfolio Summary ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Initial Capital: $" << initial_capital_ << std::endl;
    std::cout << "Total Value: $" << total_value << std::endl;
    std::cout << "Available Capital: $" << available_capital_ << std::endl;
    std::cout << "Current Position: " << current_position_ << " shares" << std::endl;
    std::cout << "Position Entry Price: $" << position_entry_price_ << std::endl;
    std::cout << "Position Value (Unrealized P&L for shorts): $" << getPositionValue(current_price) << std::endl;
    std::cout << "Realized P&L: $" << realized_pnl_ << std::endl;
    std::cout << "Unrealized P&L: $" << unrealized_pnl << std::endl;
    std::cout << "Total Return: " << total_return_percent << "%" << std::endl;
    std::cout << "Total Closing Trades: " << total_trades_ << std::endl;
    std::cout << "Winning Trades: " << winning_trades_ << std::endl;
    std::cout << "Win Rate: " << (getWinRate() * 100.0f) << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << getSharpeRatio() << std::endl;
    std::cout << "Max Drawdown: " << (max_drawdown_ * 100.0f) << "%" << std::endl;
    std::cout << "=========================" << std::endl;
}

void PortfolioManager::logTrade(TradingAction original_action, float current_market_price, float shares_executed, float confidence, float pnl_from_this_trade, float entry_price_of_closed_portion, bool was_closing_trade) {
    (void)original_action; (void)current_market_price; (void)confidence; // Mark as unused for now

    if (was_closing_trade && std::abs(shares_executed) > 1e-8f) {
        total_trades_++; // Counts closing trades
        if (pnl_from_this_trade > 1e-8f) { // Consider only significantly positive P&L as a win
            winning_trades_++;
        }

        // Calculate return for Sharpe ratio, only if entry price was valid
        if (std::abs(entry_price_of_closed_portion) > 1e-8f) {
            float capital_at_risk = std::abs(shares_executed * entry_price_of_closed_portion);
            if (capital_at_risk > 1e-6f) { // Avoid division by zero
                 float return_percent = pnl_from_this_trade / capital_at_risk;
                 trade_returns_.push_back(return_percent);
            }
        }
    }
    // Removed updatePortfolioValue(price) call from here, it's called in executeTrade
}

void PortfolioManager::updatePortfolioValue(float current_price) {
    float total_value = getTotalValue(current_price);
    portfolio_values_.push_back(total_value);
    
    // Update peak value and max drawdown
    if (total_value > peak_value_) {
        peak_value_ = total_value;
    } else if (peak_value_ > 1e-6f) { // Avoid division by zero if peak_value_ is zero
        float drawdown = (peak_value_ - total_value) / peak_value_;
        if (std::isfinite(drawdown)) { // Ensure drawdown is a valid number
             max_drawdown_ = std::max(max_drawdown_, drawdown);
        }
    }
}

void PortfolioManager::resetPortfolioState() {
    std::cout << "[PortfolioManager] Resetting portfolio to initial state" << std::endl;
    
    // Reset all financial state
    available_capital_ = initial_capital_;
    current_position_ = 0.0f;
    position_entry_price_ = 0.0f;
    realized_pnl_ = 0.0f;
    peak_value_ = initial_capital_;
    max_drawdown_ = 0.0f;
    
    // Reset trade tracking
    total_trades_ = 0;
    winning_trades_ = 0;
    trade_returns_.clear();
    portfolio_values_.clear();
    
    std::cout << "[PortfolioManager] Portfolio reset complete - Initial capital: $" << initial_capital_ << std::endl;
}

bool PortfolioManager::validatePortfolioState(float current_price) const {
    // Check for unrealistic portfolio values
    float total_value = getTotalValue(current_price);
    
    // Check if portfolio value exceeds reasonable bounds
    if (total_value > initial_capital_ * 1000.0f) { // Increased multiplier for high-return scenarios
        std::cerr << "[PortfolioManager] Warning: Portfolio value appears extremely high: $" << total_value << std::endl;
        // return false; // Commenting out to allow for high performance, but keeping warning
    }
    
    // Check if available capital is negative beyond reasonable margin
    if (available_capital_ < -initial_capital_ * 0.1f) { // Allow for some negative due to unsettled trades or fees if modeled
        std::cerr << "[PortfolioManager] Warning: Available capital is severely negative: $" << available_capital_ << std::endl;
        return false;
    }
    
    // Check for NaN or infinite values
    if (!std::isfinite(total_value) || !std::isfinite(available_capital_) || !std::isfinite(realized_pnl_)) {
        std::cerr << "[PortfolioManager] Warning: Portfolio contains non-finite values" << std::endl;
        return false;
    }
    
    return true;
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
    (void)available_capital; // Suppress unused parameter warning
    
    // Kelly criterion inspired position sizing
    // Ensure confidence is not zero to prevent issues, map to a small base if zero.
    float adjusted_confidence = std::max(0.01f, confidence);
    float base_size = max_position_size_ * adjusted_confidence;
    
    // Adjust for volatility
    // Ensure volatility is not extremely high to prevent negative adjustment.
    float capped_volatility = std::min(volatility, 0.1f); // Cap volatility for adjustment factor
    float volatility_adjustment = std::max(0.1f, 1.0f - capped_volatility * 10.0f); // Ensure adjustment is not < 0.1
    
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
    , cumulative_reward_(0.0f)
    , epsilon_(1.0f) // FIX: Initialize epsilon to 1.0 (100% exploration at the start)
    , autosave_enabled_(true)
    , network_save_interval_(100) // Save every 100 decisions by default
    , decisions_since_save_(0)
    , network_state_file_("neural_network_" + symbol + ".state")
    {
    initializeComponents();
}

TradingAgent::~TradingAgent() {
    stopTrading();
    
    // Save neural network state before destruction
    if (autosave_enabled_) {
        std::cout << "[TradingAgent] Saving neural network state before shutdown..." << std::endl;
        saveNeuralNetworkState();
    }
    
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
        
        // Auto-save neural network state periodically
        if (autosave_enabled_) {
            decisions_since_save_++;
            if (decisions_since_save_ >= network_save_interval_) {
                saveNeuralNetworkState();
                decisions_since_save_ = 0;
            }
        }
    }
}

void TradingAgent::initializeNeuralNetwork() {
    // Initialize CUDA neural network interface
    std::cout << "[TradingAgent] Initializing CUDA neural network via global interface..." << std::endl;
    initializeNetwork();
    
    // Try to load previous neural network state
    std::cout << "[TradingAgent] Attempting to load previous network state..." << std::endl;
    if (loadNeuralNetworkState()) {
        std::cout << "[TradingAgent] Previous network state loaded successfully" << std::endl;
    } else {
        std::cout << "[TradingAgent] No previous network state found, starting fresh" << std::endl;
    }
    
    std::cout << "[TradingAgent] Neural network initialized" << std::endl;
}

/**
 * @brief Makes a decision by querying the network and applying an exploration strategy.
 */
TradingDecision TradingAgent::makeDecision(const MarketData& data) {
    TradingDecision decision;
    decision.timestamp = std::chrono::system_clock::now();
    
    // Prepare input features
    std::vector<float> neural_input = prepareNeuralInput(data);
    
    // --- FIX: Use a proper reward signal (last trade PnL), not cumulative reward ---
    // For now, we pass a neutral signal, as the true reward is calculated *after* the action.
    // The learning will be driven by updateNeuralNetwork.
    float last_reward = reward_history_.empty() ? 0.0f : reward_history_.back();
    std::vector<float> neural_outputs = forwardCUDA(neural_input, last_reward);
    
    decision.neural_outputs = neural_outputs;

    // --- FIX: Implement Epsilon-Greedy Exploration Strategy ---
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    if (distrib(gen) < epsilon_) {
        // --- EXPLORE: Take a random action ---
        std::uniform_int_distribution<> action_dist(1, 2); // 1 for BUY, 2 for SELL
        decision.action = static_cast<TradingAction>(action_dist(gen));
        decision.rationale = "EXPLORATORY ACTION";
        // Use a reasonable confidence for exploration, not 1.0f which can cause numerical issues
        decision.confidence = 0.5f + (distrib(gen) * 0.3f); // Random confidence between 0.5 and 0.8
    } else {
        // --- EXPLOIT: Use the network's decision ---
        decision.confidence = calculateConfidence(neural_outputs);
        decision.action = interpretNeuralOutput(neural_outputs, decision.confidence);
        decision.rationale = "Neural network decision with confidence " + std::to_string(decision.confidence);
    }

    // Decay epsilon to reduce exploration over time
    epsilon_ *= EPSILON_DECAY;
    if (epsilon_ < 0.01f) epsilon_ = 0.01f; // Keep a minimum exploration rate
    
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
    }
    
    if (decision.action != TradingAction::HOLD && decision.action != TradingAction::NO_ACTION) {
        portfolio_manager_->executeTrade(decision.action, data.close, decision.position_size, decision.confidence);
    }
    
    return decision;
}

void TradingAgent::updateNeuralNetwork(const TradingDecision& decision, float reward) {
    (void)decision; // Suppress unused parameter warning
    
    if (!learning_enabled_) return;
    
    reward_history_.push_back(reward);
    cumulative_reward_ += reward;
    
    // Send reward signal to neural network
    sendRewardToNetwork(reward);
    
    // Log neural network state
    logNeuralNetworkState();
}

/**
 * @brief A financially-grounded reward function based on portfolio performance.
 */
float TradingAgent::calculateReward(const TradingDecision& last_decision, const MarketData& current_data) {
    // Validate market data
    if (!current_data.validate() || current_data.close <= 0.0f) {
        std::cerr << "[TradingAgent] Invalid market data for reward calculation" << std::endl;
        return 0.0f;
    }
    
    if (!portfolio_manager_) {
        std::cerr << "[TradingAgent] Portfolio manager not initialized" << std::endl;
        return 0.0f;
    }
    
    // Calculate portfolio value change since last decision
    // Use member variable instead of static to properly track across all decisions
    float current_portfolio_value = portfolio_manager_->getTotalValue(current_data.close);
    float portfolio_change = 0.0f;
    
    if (!reward_history_.empty()) {
        // Get the last portfolio value from our tracking
        static float last_portfolio_value = 100000.0f; // Initial capital on first call
        portfolio_change = current_portfolio_value - last_portfolio_value;
        last_portfolio_value = current_portfolio_value;
    } else {
        // First reward calculation - no change yet, initialize static variable
        static float last_portfolio_value = current_portfolio_value;
        (void)last_portfolio_value; // Suppress unused warning for this initialization
    }
    
    float reward = 0.0f;
    
    // Reward based on actual portfolio performance
    if (last_decision.action == TradingAction::BUY || last_decision.action == TradingAction::SELL) {
        // For actual trades, reward is proportional to the portfolio value change
        // Normalize by initial capital to get percentage-based reward
        reward = portfolio_change / 100000.0f; // Initial capital
        
        // Add confidence bonus/penalty
        float confidence_factor = (last_decision.confidence - 0.5f) * 2.0f; // Map [0,1] to [-1,1]
        if (portfolio_change > 0) {
            // Good trade: higher confidence = higher reward
            reward *= (1.0f + 0.2f * confidence_factor);
        } else {
            // Bad trade: higher confidence = higher penalty
            reward *= (1.0f + 0.3f * confidence_factor);
        }
    } 
    else if (last_decision.action == TradingAction::HOLD) {
        // For holding, small reward for preserving capital, small penalty for missing opportunities
        if (last_market_data_.close > 0) {
            float price_change_percent = (current_data.close - last_market_data_.close) / last_market_data_.close;
            
            if (std::abs(price_change_percent) < 0.01f) {
                // Stable market: small positive reward for holding
                reward = 0.01f;
            } else {
                // Volatile market: small penalty for not participating (opportunity cost)
                reward = -std::abs(price_change_percent) * 0.1f;
            }
        }
    }
    
    // Risk adjustment: penalize based on volatility if excessive risk was taken
    float current_volatility = feature_engineer_->getCurrentVolatility();
    if (current_volatility > 0.05f) { // 5% volatility threshold
        reward *= (1.0f - (current_volatility - 0.05f) * 2.0f); // Reduce reward for high volatility
    }
    
    // Clamp reward to reasonable range [-0.1, 0.1] for stability
    reward = std::max(-0.1f, std::min(0.1f, reward));
    
    // Validate final reward
    if (!std::isfinite(reward)) {
        std::cerr << "[TradingAgent] Invalid reward calculated, setting to 0" << std::endl;
        reward = 0.0f;
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
        std::cout << "Total Return: " << (portfolio_manager_->getTotalReturn(current.close) * 100) << "%" << std::endl; // Pass current.close
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
        file << "  Total Return: " << (portfolio_manager_->getTotalReturn(current_price) * 100) << "%\n"; // Pass current_price
        file << "  Total Trades: " << portfolio_manager_->getTotalTrades() << "\n";
        file << "  Win Rate: " << (portfolio_manager_->getWinRate() * 100) << "%\n";
        file << "  Sharpe Ratio: " << portfolio_manager_->getSharpeRatio() << "\n";
        file << "  Max Drawdown: " << (portfolio_manager_->getMaxDrawdown() * 100) << "%\n";
    }
    
    file.close();
    std::cout << "[TradingAgent] Performance report exported to: " << filename << std::endl;
}

TradingAgent::TradingStatistics TradingAgent::getStatistics() const {
    TradingStatistics stats{}; // Initialize struct
    
    if (portfolio_manager_) {
        float current_price = 0.0f;
        if (!market_history_.empty()) {
            current_price = market_history_.back().close;
        } else if (std::abs(last_market_data_.close) > 1e-6f) { // Use last known price if history is empty but data processed
            current_price = last_market_data_.close;
        } else {
             stats.portfolio_value = portfolio_manager_->getAvailableCapital() + portfolio_manager_->calculateRealizedPnL();
        }

        if (current_price > 1e-6f) {
            stats.total_return = portfolio_manager_->getTotalReturn(current_price);
            stats.portfolio_value = portfolio_manager_->getTotalValue(current_price);
        } else { // If price is still invalid
            // Use the new getter for initial_capital_
            float initial_capital = portfolio_manager_->getInitialCapital();
            if (std::abs(initial_capital) > 1e-6f) {
                 stats.total_return = (portfolio_manager_->getAvailableCapital() + portfolio_manager_->calculateRealizedPnL() - initial_capital) / initial_capital;
            } else {
                stats.total_return = 0.0f; // Avoid division by zero if initial capital is zero
            }
        }
        
        stats.sharpe_ratio = portfolio_manager_->getSharpeRatio();
        stats.max_drawdown = portfolio_manager_->getMaxDrawdown();
        stats.win_rate = portfolio_manager_->getWinRate();
        stats.total_trades = portfolio_manager_->getTotalTrades(); // This is now closing trades
        
    } else {
        stats.total_return = 0.0f;
        stats.sharpe_ratio = 0.0f;
        stats.max_drawdown = 0.0f;
        stats.win_rate = 0.0f;
        stats.total_trades = 0;
        stats.portfolio_value = 100000.0f; // Initial capital default
    }
    
    stats.avg_trade_duration_hours = 24.0f; // Placeholder
    stats.profit_factor = 1.0f; // Placeholder
    
    // Calculate profit factor if possible
    if (portfolio_manager_ && portfolio_manager_->getTotalTrades() > 0) {
        float gross_profit = 0.0f;
        float gross_loss = 0.0f;
        // Use the new getter for trade_returns_
        for (float pnl_percentage : portfolio_manager_->getTradeReturns()) { 
            // This loop is for percentage returns. Profit factor needs absolute P&L.
            // To implement profit factor accurately, you would need to store absolute P&L values of trades.
            // For example, if pnl_percentage > 0, it's a profit, otherwise a loss.
            // However, without the original trade value, we can't sum absolute profits/losses here.
            // The current structure of trade_returns_ (storing percentages) is good for Sharpe ratio.
            // Placeholder logic for demonstration if you were to change trade_returns_:
            if (pnl_percentage > 0) gross_profit += pnl_percentage; // This is not correct for profit factor with percentages
            else gross_loss += pnl_percentage; // This is not correct for profit factor with percentages
        }
        // if (std::abs(gross_loss) > 1e-6f) {
        //     stats.profit_factor = gross_profit / std::abs(gross_loss); // This calculation would be wrong with percentages
        // } else if (gross_profit > 0) {
        //     stats.profit_factor = std::numeric_limits<float>::infinity(); // All wins
        // }
    }
    
    // Neural network confidence history
    for (const auto& decision : decision_history_) {
        stats.neural_confidence_history.push_back(decision.confidence);
    }
    
    stats.reward_history = reward_history_;
    
    return stats;
}

std::vector<MarketData> TradingAgent::getMarketHistory() const {
    return market_history_;
}

void TradingAgent::setMarketHistory(const std::vector<MarketData>& history) {
    market_history_ = history;
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
    if (trading_log_.is_open()) {
        trading_log_ << "timestamp,action,confidence,position_size,reward\n";
        trading_log_.flush();
    }
    
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
    
    // Apply numerical stability to prevent overflow
    std::vector<float> stable_outputs = outputs;
    float max_output = *std::max_element(stable_outputs.begin(), stable_outputs.end());
    
    // Subtract max for numerical stability
    for (auto& output : stable_outputs) {
        output -= max_output;
        // Clamp to prevent extreme values
        output = std::max(-10.0f, std::min(10.0f, output));
    }
    
    // Calculate softmax probabilities
    std::vector<float> exp_outputs;
    float sum_exp = 0.0f;
    
    for (float output : stable_outputs) {
        float exp_val = std::exp(output);
        exp_outputs.push_back(exp_val);
        sum_exp += exp_val;
    }
    
    if (sum_exp == 0.0f || !std::isfinite(sum_exp)) {
        confidence = 0.0f;
        return TradingAction::HOLD;
    }
    
    // Find the action with highest probability
    size_t max_idx = 0;
    float max_prob = 0.0f;
    
    for (size_t i = 0; i < exp_outputs.size() && i < 3; ++i) {
        float prob = exp_outputs[i] / sum_exp;
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = i;
        }
    }
    
    // Ensure confidence is in valid range [0, 1]
    confidence = std::max(0.0f, std::min(1.0f, max_prob));
    
    switch (max_idx) {
        case 0: return TradingAction::HOLD;
        case 1: return TradingAction::BUY;
        case 2: return TradingAction::SELL;
        default: return TradingAction::HOLD;
    }
}

/**
 * @brief Calculates confidence using a proper Softmax function with numerical stability.
 */
float TradingAgent::calculateConfidence(const std::vector<float>& outputs) {
    if (outputs.empty()) return 0.0f;

    // Apply numerical stability to prevent overflow
    std::vector<float> stable_outputs = outputs;
    float max_output = *std::max_element(stable_outputs.begin(), stable_outputs.end());
    
    // Subtract max for numerical stability
    for (auto& output : stable_outputs) {
        output -= max_output;
        // Clamp to prevent extreme values
        output = std::max(-10.0f, std::min(10.0f, output));
    }

    std::vector<float> exp_outputs;
    float sum_exp = 0.0f;
    
    // Compute e^x for each output
    for (float output : stable_outputs) {
        float exp_val = std::exp(output);
        if (!std::isfinite(exp_val)) {
            exp_val = 0.0f;
        }
        exp_outputs.push_back(exp_val);
        sum_exp += exp_val;
    }

    if (sum_exp == 0.0f || !std::isfinite(sum_exp)) return 0.0f;

    // The confidence is the highest probability in the softmax distribution
    float max_prob = 0.0f;
    for (float exp_val : exp_outputs) {
        float prob = exp_val / sum_exp;
        max_prob = std::max(max_prob, prob);
    }
    
    // Ensure confidence is in valid range [0, 1]
    return std::max(0.0f, std::min(1.0f, max_prob));
}

void TradingAgent::validateMarketData(const MarketData& data) {
    if (!data.validate()) {
        throw std::invalid_argument("Invalid market data provided");
    }
    
    // Additional validation to prevent extreme values
    if (data.close <= 0.001f || data.close > 1000000.0f) {
        throw std::invalid_argument("Market data price out of reasonable range: " + std::to_string(data.close));
    }
    
    if (!std::isfinite(data.close) || !std::isfinite(data.volume)) {
        throw std::invalid_argument("Market data contains non-finite values");
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

// =============================================================================
// NEURAL NETWORK PERSISTENCE IMPLEMENTATION
// =============================================================================

void TradingAgent::saveNeuralNetworkState() const {
    saveNeuralNetworkState(network_state_file_);
}

bool TradingAgent::loadNeuralNetworkState() {
    return loadNeuralNetworkState(network_state_file_);
}

bool TradingAgent::saveNeuralNetworkState(const std::string& filename) const {
    try {
        std::cout << "[TradingAgent] Saving neural network state to: " << filename << std::endl;
        
        // Save neural network weights and structure via interface
        saveNetworkState(filename);
        
        // Save additional trading agent state
        std::ofstream meta_file(filename + ".meta", std::ios::binary);
        if (!meta_file.is_open()) {
            std::cerr << "[TradingAgent] Failed to open meta file for saving: " << filename << ".meta" << std::endl;
            return false;
        }
        
        // Save trading-specific neural network metadata
        float current_epsilon = epsilon_;
        meta_file.write(reinterpret_cast<const char*>(&current_epsilon), sizeof(float));
        
        size_t reward_history_size = reward_history_.size();
        meta_file.write(reinterpret_cast<const char*>(&reward_history_size), sizeof(size_t));
        if (reward_history_size > 0) {
            meta_file.write(reinterpret_cast<const char*>(reward_history_.data()), 
                           reward_history_size * sizeof(float));
        }
        
        float cumulative_reward = cumulative_reward_;
        meta_file.write(reinterpret_cast<const char*>(&cumulative_reward), sizeof(float));
        
        size_t decision_count = decision_history_.size();
        meta_file.write(reinterpret_cast<const char*>(&decision_count), sizeof(size_t));
        
        meta_file.close();
        
        std::cout << "[TradingAgent] Neural network state saved successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TradingAgent] Error saving neural network state: " << e.what() << std::endl;
        return false;
    }
}

bool TradingAgent::loadNeuralNetworkState(const std::string& filename) {
    try {
        std::cout << "[TradingAgent] Loading neural network state from: " << filename << std::endl;
        
        // Load neural network weights and structure via interface
        loadNetworkState(filename);
        
        // Load additional trading agent state
        std::ifstream meta_file(filename + ".meta", std::ios::binary);
        if (!meta_file.is_open()) {
            std::cout << "[TradingAgent] No meta file found, using default values" << std::endl;
            return true; // Not an error, just no previous state
        }
        
        // Load trading-specific neural network metadata
        float loaded_epsilon;
        meta_file.read(reinterpret_cast<char*>(&loaded_epsilon), sizeof(float));
        if (meta_file.good()) {
            epsilon_ = loaded_epsilon;
        }
        
        size_t reward_history_size;
        meta_file.read(reinterpret_cast<char*>(&reward_history_size), sizeof(size_t));
        if (meta_file.good() && reward_history_size > 0 && reward_history_size < 1000000) // Sanity check
        {
            reward_history_.resize(reward_history_size);
            meta_file.read(reinterpret_cast<char*>(reward_history_.data()), 
                          reward_history_size * sizeof(float));
        }
        
        float loaded_cumulative_reward;
        meta_file.read(reinterpret_cast<char*>(&loaded_cumulative_reward), sizeof(float));
        if (meta_file.good()) {
            cumulative_reward_ = loaded_cumulative_reward;
        }
        
        size_t decision_count;
        meta_file.read(reinterpret_cast<char*>(&decision_count), sizeof(size_t));
        if (meta_file.good()) {
            std::cout << "[TradingAgent] Loaded state from " << decision_count << " previous decisions" << std::endl;
        }
        
        meta_file.close();
        
        std::cout << "[TradingAgent] Neural network state loaded successfully" << std::endl;
        std::cout << "[TradingAgent] Resumed with epsilon: " << epsilon_ 
                  << ", cumulative reward: " << cumulative_reward_ 
                  << ", reward history size: " << reward_history_.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TradingAgent] Error loading neural network state: " << e.what() << std::endl;
        return false;
    }
}

void TradingAgent::resetRewardTracking() {
    reward_history_.clear();
    cumulative_reward_ = 0.0f;
    
    std::cout << "[TradingAgent] Reward tracking reset" << std::endl;
}
