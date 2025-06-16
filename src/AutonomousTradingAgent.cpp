#include "AutonomousTradingAgent.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <nlohmann/json.hpp> // Added for JSON operations
#include <sstream>      // For std::stringstream, already in .h but good practice

// AutonomousTradingAgent::AutonomousTradingAgent()
//     : symbol_(""), 
//       coinbase_api_ptr_(nullptr), 
//       portfolio_(0.0), 
//       tech_analyzer_(price_series_), 
//       max_price_series_size_(200), 
//       cumulative_reward_(0.0),
//       epsilon_(1.0), 
//       is_initialized_(false)
//        {
//     log_file_.open("autonomous_trading_agent.log");
//     if (!log_file_.is_open()) {
//         std::cerr << "[AutonomousTradingAgent] Warning: Could not open log file" << std::endl;
//     }
//     std::cout << "[AutonomousTradingAgent] Agent default-initialized" << std::endl;
// }

AutonomousTradingAgent::AutonomousTradingAgent(CoinbaseAdvancedTradeApi* api_client)
    : symbol_(""),
      coinbase_api_ptr_(api_client),
      portfolio_(0.0),
      tech_analyzer_(price_series_),
      max_price_series_size_(200),
      cumulative_reward_(0.0),
      epsilon_(1.0),
      is_initialized_(false)
{
    log_file_.open("autonomous_trading_agent.log");
    if (!log_file_.is_open()) {
        std::cerr << "[AutonomousTradingAgent] Warning: Could not open log file" << std::endl;
    }
    if (coinbase_api_ptr_) {
        std::cout << "[AutonomousTradingAgent] Agent initialized with API client." << std::endl;
    } else {
        std::cout << "[AutonomousTradingAgent] Agent initialized without API client (CSV mode)." << std::endl;
    }
}

AutonomousTradingAgent::~AutonomousTradingAgent() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
    
    std::cout << "[AutonomousTradingAgent] Agent destroyed" << std::endl;
}

// bool AutonomousTradingAgent::initialize(const std::string& symbol, BinanceApi& binance_api, double initial_cash) {
//     symbol_ = symbol;
//     binance_api_ptr_ = &binance_api; 
//     portfolio_.reset(initial_cash, 0.0); 

//     if (!nn_interface_.initialize()) {
//         std::cerr << "[AutonomousTradingAgent] CRITICAL: Neural Network Interface failed to initialize for symbol " << symbol_ << std::endl;
//         return false; // Return false if NN initialization fails
//     }

//     price_series_.clear(); // Clear any old data
//     // tech_analyzer_.updatePriceSeries(price_series_); // Done in constructor or explicitly after series modification

//     cumulative_reward_ = 0.0;
//     epsilon_ = 1.0; 
//     decision_history_.clear();
//     reward_history_.clear();
//     is_initialized_ = true; // Set to true after successful initialization steps

//     std::cout << "[AutonomousTradingAgent] Agent initialized for " << symbol_ 
//               << " with initial cash: $" << std::fixed << std::setprecision(2) << initial_cash << std::endl;
//     log_file_ << "Agent initialized for " << symbol_ << " with initial cash: " << initial_cash << std::endl;
    
//     return true; // Return true on successful initialization
// }

// bool AutonomousTradingAgent::initialize(const std::string& symbol, CoinbaseAdvancedTradeApi& coinbase_api, double initial_cash) { // Old signature
bool AutonomousTradingAgent::initialize(const std::string& symbol, double initial_cash, CoinbaseAdvancedTradeApi* api_client) { // New signature
    symbol_ = symbol;
    coinbase_api_ptr_ = api_client; // Store the pointer
    portfolio_.reset(initial_cash, 0.0);

    if (!nn_interface_.initialize()) {
        std::cerr << "[AutonomousTradingAgent] CRITICAL: Neural Network Interface failed to initialize for symbol " << symbol_ << std::endl;
        return false;
    }

    price_series_.clear();
    cumulative_reward_ = 0.0;
    epsilon_ = 1.0;
    decision_history_.clear();
    reward_history_.clear();
    is_initialized_ = true;

    std::cout << "[AutonomousTradingAgent] Agent initialized for " << symbol_
              << " with initial cash: $" << std::fixed << std::setprecision(2) << initial_cash << std::endl;
    log_file_ << "Agent initialized for " << symbol_ << " with initial cash: " << initial_cash;
    if (coinbase_api_ptr_) {
        log_file_ << " (API client connected)" << std::endl;
        std::cout << "[AutonomousTradingAgent] API client is connected." << std::endl;
    } else {
        log_file_ << " (API client is NULL - CSV mode)" << std::endl;
        std::cout << "[AutonomousTradingAgent] API client is NULL (CSV mode)." << std::endl;
    }

    return true;
}

AutonomousTradingAgent::TradingDecision AutonomousTradingAgent::makeDecision(int current_tick_index, double current_price) {
    std::cout << "[Agent] Making decision for tick " << current_tick_index << " at price $" << current_price << std::endl;
    if (!is_initialized_) { // Use the member variable
        std::cerr << "[Agent] Error: Agent not initialized." << std::endl;
        return TradingDecision::HOLD;
    }

    // Construct a PriceTick (assuming current_price is 'close', others might be needed or set to current_price)
    // This is a simplification. Ideally, the full PriceTick comes from Simulation.
    PriceTick current_tick_data;
    current_tick_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); // Example timestamp
    current_tick_data.open = current_price; // Simplification
    current_tick_data.high = current_price; // Simplification
    current_tick_data.low = current_price;  // Simplification
    current_tick_data.close = current_price;
    current_tick_data.volume = 0; // Simplification, volume data might be important

    appendPriceTick(current_tick_data); // Update internal price series and TA

    std::map<std::string, double> features;
    if (nn_interface_.isInitialized()) { // Only get features if NN is ready
        features = _getCurrentMarketFeatures(current_tick_index, current_price);
        if (features.empty()) {
            std::cout << "[Agent] Warning: No features generated for NN." << std::endl;
        }
    } else {
        std::cout << "[Agent] Info: Neural network not initialized. Decision will be random or default." << std::endl;
    }

    TradingDecision decision = TradingDecision::HOLD;
    double confidence = 0.0; // Confidence in the decision, can be set by NN or heuristics

    // Exploration vs. Exploitation (Epsilon-Greedy)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (nn_interface_.isInitialized() && !features.empty()) {
        std::cout << "[Agent] NN is initialized and features are available." << std::endl;
    } else if (!nn_interface_.isInitialized()){
        std::cout << "[Agent] NN is NOT initialized. Proceeding with random action." << std::endl;
    } else if (features.empty()){
        std::cout << "[Agent] NN is initialized but features are EMPTY. Proceeding with random action." << std::endl;
    }


    if (!nn_interface_.isInitialized() || features.empty() || dis(gen) < epsilon_) {
        // Explore: Choose a random action
        std::uniform_int_distribution<> action_dist(0, 2); // 0: BUY, 1: SELL, 2: HOLD
        int random_action = action_dist(gen);
        decision = static_cast<TradingDecision>(random_action);
        confidence = 0.5; // Default confidence for random actions

        if (log_file_.is_open()) {
            log_file_ << decision_log_buffer_.rdbuf(); // Log any previous decision details
            decision_log_buffer_.str(""); // Clear buffer
            log_file_ << "[Decision Log] Tick: " << current_tick_index << ", Price: " << current_price
                      << (nn_interface_.isInitialized() && !features.empty() ? ", Strategy: Exploration (epsilon)" : ", Strategy: Exploration (NN not ready/no features)")
                      << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                      << ", Confidence: " << confidence;
        }
         std::cout << "[Agent] " << (nn_interface_.isInitialized() && !features.empty() ? "Exploration (epsilon):" : "Exploration (NN not ready/no features):")
                  << " Random action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD")) << std::endl;

    } else {
        // Exploit: Use Neural Network prediction
        std::vector<double> prediction_signal_vector = nn_interface_.getPrediction(features);
        double prediction_signal = 0.0; // Default to neutral if prediction is empty or invalid
        if (!prediction_signal_vector.empty()) {
            prediction_signal = prediction_signal_vector[0]; // Assuming the first element is the primary signal
            // Potentially, other elements could be confidence, or signals for other actions
            if (prediction_signal_vector.size() > 1) {
                 confidence = prediction_signal_vector[1]; // Example: second element is confidence
            } else {
                confidence = std::abs(prediction_signal); // Or derive confidence from signal strength
            }
        } else {
            std::cout << "[Agent] Warning: NN returned empty prediction. Defaulting to HOLD." << std::endl;
            // decision remains HOLD, confidence remains 0.0 or could be set to a low value
        }
        
        decision = _determineActionFromSignal(prediction_signal, confidence);

        if (log_file_.is_open()) {
            log_file_ << decision_log_buffer_.rdbuf();
            decision_log_buffer_.str("");
            log_file_ << "[Decision Log] Tick: " << current_tick_index << ", Price: " << current_price
                      << ", Strategy: Exploitation (NN)"
                      << ", NN Signal: " << prediction_signal 
                      << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                      << ", Confidence: " << confidence;
        }
        std::cout << "[Agent] Exploitation (NN): Signal=" << prediction_signal 
                  << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                  << ", Confidence: " << confidence << std::endl;
    }

    double quantity_to_trade = _determineQuantity(current_price, decision, confidence);
    bool trade_executed = false;

    if (quantity_to_trade > 0) {
        if (decision == TradingDecision::BUY) {
            trade_executed = portfolio_.executeBuy(quantity_to_trade, current_price); // Remove symbol_
        } else if (decision == TradingDecision::SELL) {
            trade_executed = portfolio_.executeSell(quantity_to_trade, current_price); // Remove symbol_
        }
    }

    if (quantity_to_trade > 0 && !trade_executed) {
        std::cout << "[Agent] Trade for " << quantity_to_trade << " units failed. Reverting to HOLD." << std::endl;
        decision = TradingDecision::HOLD; // Revert to HOLD if trade failed
        quantity_to_trade = 0; // No quantity traded
    }


    DecisionRecord current_decision_record;
    current_decision_record.tick_index = current_tick_index;
    current_decision_record.timestamp = std::chrono::system_clock::now(); // Correct timestamp assignment
    current_decision_record.price_at_decision = current_price;
    current_decision_record.decision = decision;
    current_decision_record.confidence = confidence;
    current_decision_record.quantity = quantity_to_trade; 
    current_decision_record.portfolio_value_before = portfolio_.getCurrentValue(current_price); 
    current_decision_record.reward_after = 0.0; // Initialize reward_after

    decision_history_.push_back(current_decision_record);
    _logDecision(current_decision_record); // Removed reward from call

    if (log_file_.is_open()) {
        log_file_ << ", Quantity Traded: " << quantity_to_trade
                  << ", Portfolio Value: " << portfolio_.getCurrentValue(current_price)
                  << ", Cash: " << portfolio_.getCashBalance()
                  << ", Coins: " << portfolio_.getCoinBalance() << std::endl;
    }

    return decision;
}

AutonomousTradingAgent::TradingDecision AutonomousTradingAgent::makeDecision(int current_tick_index, const PriceTick& price_tick) {
    std::cout << "[Agent] Making decision for tick " << current_tick_index << " using full OHLCV data" << std::endl;
    std::cout << "[Agent] Price data: O=" << price_tick.open << " H=" << price_tick.high 
              << " L=" << price_tick.low << " C=" << price_tick.close << " V=" << price_tick.volume << std::endl;
    
    if (!is_initialized_) {
        std::cerr << "[Agent] Error: Agent not initialized." << std::endl;
        return TradingDecision::HOLD;
    }

    // Use the complete price tick data
    appendPriceTick(price_tick);
    
    double current_price = price_tick.close; // Use closing price for decision making
    
    std::map<std::string, double> features;
    if (nn_interface_.isInitialized()) {
        features = _getCurrentMarketFeatures(current_tick_index, current_price);
        if (features.empty()) {
            std::cout << "[Agent] Warning: No features generated for NN." << std::endl;
        }
    } else {
        std::cout << "[Agent] Info: Neural network not initialized. Decision will be random or default." << std::endl;
    }

    TradingDecision decision = TradingDecision::HOLD;
    double confidence = 0.0;

    // Exploration vs. Exploitation (Epsilon-Greedy)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (!nn_interface_.isInitialized() || features.empty() || dis(gen) < epsilon_) {
        // Explore: Choose a random action
        std::uniform_int_distribution<> action_dist(0, 2);
        int random_action = action_dist(gen);
        decision = static_cast<TradingDecision>(random_action);
        confidence = 0.5;

        std::cout << "[Agent] Exploration: Random action: " 
                  << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD")) << std::endl;
    } else {
        // Exploit: Use Neural Network prediction
        std::vector<double> prediction_signal_vector = nn_interface_.getPrediction(features);
        double prediction_signal = 0.0;
        
        if (!prediction_signal_vector.empty()) {
            prediction_signal = prediction_signal_vector[0];
            if (prediction_signal_vector.size() > 1) {
                confidence = prediction_signal_vector[1];
            } else {
                confidence = std::abs(prediction_signal);
            }
        } else {
            std::cout << "[Agent] Warning: NN returned empty prediction. Defaulting to HOLD." << std::endl;
        }
        
        decision = _determineActionFromSignal(prediction_signal, confidence);
        
        std::cout << "[Agent] Exploitation (NN): Signal=" << prediction_signal 
                  << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                  << ", Confidence: " << confidence << std::endl;
    }

    double quantity_to_trade = _determineQuantity(current_price, decision, confidence);
    bool trade_executed = false;

    if (quantity_to_trade > 0) {
        if (decision == TradingDecision::BUY) {
            trade_executed = portfolio_.executeBuy(quantity_to_trade, current_price);
        } else if (decision == TradingDecision::SELL) {
            trade_executed = portfolio_.executeSell(quantity_to_trade, current_price);
        }
    }

    if (quantity_to_trade > 0 && !trade_executed) {
        std::cout << "[Agent] Trade for " << quantity_to_trade << " units failed. Reverting to HOLD." << std::endl;
        decision = TradingDecision::HOLD;
        quantity_to_trade = 0;
    }

    // Apply epsilon decay
    if (epsilon_ > 0.01) { // Minimum epsilon to maintain some exploration
        epsilon_ *= EPSILON_DECAY;
    }

    DecisionRecord current_decision_record;
    current_decision_record.tick_index = current_tick_index;
    current_decision_record.timestamp = std::chrono::system_clock::now();
    current_decision_record.price_at_decision = current_price;
    current_decision_record.decision = decision;
    current_decision_record.confidence = confidence;
    current_decision_record.quantity = quantity_to_trade;
    current_decision_record.portfolio_value_before = portfolio_.getCurrentValue(current_price);
    current_decision_record.reward_after = 0.0;

    decision_history_.push_back(current_decision_record);
    _logDecision(current_decision_record);

    if (log_file_.is_open()) {
        log_file_ << "[Full OHLCV Decision] Tick: " << current_tick_index 
                  << ", OHLCV: " << price_tick.open << "/" << price_tick.high << "/" << price_tick.low << "/" << price_tick.close << "/" << price_tick.volume
                  << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                  << ", Quantity: " << quantity_to_trade
                  << ", Portfolio Value: " << portfolio_.getCurrentValue(current_price)
                  << ", Cash: " << portfolio_.getCashBalance()
                  << ", Coins: " << portfolio_.getCoinBalance()
                  << ", Epsilon: " << epsilon_ << std::endl;
    }

    return decision;
}

void AutonomousTradingAgent::receiveReward(double reward) {
    std::cout << "[Agent] Received reward: " << reward << std::endl;
    if (nn_interface_.isInitialized()) {
        nn_interface_.sendRewardSignal(reward);
    } else {
        std::cout << "[Agent] Info: Neural network not initialized. Reward not sent to NN." << std::endl;
    }

    // Update last decision record with the reward if history is not empty
    if (!decision_history_.empty()) {
        decision_history_.back().reward_after = reward;
    }

    if (log_file_.is_open()) {
        // Log reward against the decision that led to it.
        // This might be tricky if decisions and rewards are logged at different times.
        // For now, just appending a general reward log.
        log_file_ << "[Reward Log] Reward: " << reward << " received for previous action." << std::endl;
    }
}

// Definition for _determineActionFromSignal
AutonomousTradingAgent::TradingDecision AutonomousTradingAgent::_determineActionFromSignal(double prediction_signal, double& confidence) {
    // Basic placeholder logic
    if (prediction_signal > 0.5) { // Example threshold
        confidence = prediction_signal; // Or some other calculation
        return TradingDecision::BUY;
    } else if (prediction_signal < -0.5) { // Example threshold
        confidence = -prediction_signal; // Or some other calculation
        return TradingDecision::SELL;
    }
    confidence = 1.0 - std::abs(prediction_signal); // Example confidence for HOLD
    return TradingDecision::HOLD;
}

// Definition for _getCurrentMarketFeatures
std::map<std::string, double> AutonomousTradingAgent::_getCurrentMarketFeatures(int current_tick_index, double current_price) {
    std::map<std::string, double> features;
    
    // Basic price features
    features["current_price"] = current_price;
    features["tick_index"] = static_cast<double>(current_tick_index);

    // Add OHLCV features if we have recent price data
    if (!price_series_.empty()) {
        const PriceTick& latest_tick = price_series_.back();
        
        // OHLC features
        features["open"] = latest_tick.open;
        features["high"] = latest_tick.high;
        features["low"] = latest_tick.low;
        features["close"] = latest_tick.close;
        features["volume"] = latest_tick.volume;
        
        // Price movement features
        if (latest_tick.open > 0) {
            features["price_change_pct"] = (latest_tick.close - latest_tick.open) / latest_tick.open;
        }
        if (latest_tick.low < latest_tick.high) {
            features["wick_ratio"] = (latest_tick.high - latest_tick.low) / latest_tick.close;
            features["upper_wick"] = (latest_tick.high - std::max(latest_tick.open, latest_tick.close)) / latest_tick.close;
            features["lower_wick"] = (std::min(latest_tick.open, latest_tick.close) - latest_tick.low) / latest_tick.close;
        }
        
        // Multi-tick features if we have enough data
        if (price_series_.size() >= 2) {
            const PriceTick& prev_tick = price_series_[price_series_.size() - 2];
            if (prev_tick.close > 0) {
                features["price_momentum"] = (latest_tick.close - prev_tick.close) / prev_tick.close;
            }
            if (prev_tick.volume > 0) {
                features["volume_change"] = (latest_tick.volume - prev_tick.volume) / prev_tick.volume;
            }
        }
        
        // Calculate short-term volatility if we have enough data
        if (price_series_.size() >= 5) {
            double sum = 0.0;
            double mean = 0.0;
            int lookback = std::min(5, static_cast<int>(price_series_.size()));
            
            for (int i = price_series_.size() - lookback; i < static_cast<int>(price_series_.size()); ++i) {
                mean += price_series_[i].close;
            }
            mean /= lookback;
            
            for (int i = price_series_.size() - lookback; i < static_cast<int>(price_series_.size()); ++i) {
                double diff = price_series_[i].close - mean;
                sum += diff * diff;
            }
            features["short_volatility"] = std::sqrt(sum / lookback) / mean; // Coefficient of variation
        }
    }

    // Add technical indicators if TA is available
    if (!price_series_.empty() && current_tick_index >= 0) {
        // Use the last available index in the price series for TA calculations
        int ta_index = std::min(current_tick_index, static_cast<int>(price_series_.size()) - 1);
        
        std::map<std::string, double> ta_features = tech_analyzer_.getFeaturesForTick(ta_index);

        // Extract specific technical indicators
        if (ta_features.count("SMA_5")) features["sma5"] = ta_features["SMA_5"];
        if (ta_features.count("SMA_10")) features["sma10"] = ta_features["SMA_10"];
        if (ta_features.count("SMA_20")) features["sma20"] = ta_features["SMA_20"];
        if (ta_features.count("EMA_5")) features["ema5"] = ta_features["EMA_5"];
        if (ta_features.count("EMA_10")) features["ema10"] = ta_features["EMA_10"];
        if (ta_features.count("EMA_20")) features["ema20"] = ta_features["EMA_20"];
        
        if (ta_features.count("MACD_Line")) features["macd_line"] = ta_features["MACD_Line"];
        if (ta_features.count("MACD_Signal")) features["macd_signal"] = ta_features["MACD_Signal"];
        if (ta_features.count("MACD_Hist")) features["macd_histogram"] = ta_features["MACD_Hist"];

        if (ta_features.count("BB_Upper_20")) features["bb_upper"] = ta_features["BB_Upper_20"];
        if (ta_features.count("BB_Middle_20")) features["bb_middle"] = ta_features["BB_Middle_20"];
        if (ta_features.count("BB_Lower_20")) features["bb_lower"] = ta_features["BB_Lower_20"];

        if (ta_features.count("RSI_14")) features["rsi"] = ta_features["RSI_14"];
        if (ta_features.count("ATR_14")) features["atr"] = ta_features["ATR_14"];
        
        // Relative position within Bollinger Bands
        if (ta_features.count("BB_Upper_20") && ta_features.count("BB_Lower_20")) {
            double bb_range = ta_features["BB_Upper_20"] - ta_features["BB_Lower_20"];
            if (bb_range > 0) {
                features["bb_position"] = (current_price - ta_features["BB_Lower_20"]) / bb_range;
            }
        }

    } else if (!price_series_.empty()) {
        std::cout << "[Agent_Features] Price series available but tick index issue, series size: " << price_series_.size() << ", requested index: " << current_tick_index << std::endl;
    } else {
        std::cout << "[Agent_Features] Price series is empty, cannot calculate TA features." << std::endl;
    }

    // Portfolio-related features
    features["cash_balance"] = portfolio_.getCashBalance();
    features["coin_balance"] = portfolio_.getCoinBalance();
    features["portfolio_value"] = portfolio_.getCurrentValue(current_price);
    
    // Normalized portfolio allocation
    double total_value = portfolio_.getCurrentValue(current_price);
    if (total_value > 0) {
        features["cash_ratio"] = portfolio_.getCashBalance() / total_value;
        features["coin_ratio"] = (portfolio_.getCoinBalance() * current_price) / total_value;
    }

    std::cout << "[Agent_Features] Generated " << features.size() << " features for tick " << current_tick_index << std::endl;
    return features;
}

// Definition for _determineQuantity
double AutonomousTradingAgent::_determineQuantity(double current_price, TradingDecision decision, double confidence) {
    // Basic placeholder logic: trade a fixed percentage of cash for BUYs, or a fixed percentage of coins for SELLs
    // Ensure current_price is not zero to avoid division by zero
    if (current_price <= 0) return 0.0;

    double quantity = 0.0;
    const double trade_fraction = 0.1; // Trade 10% of available resource, scaled by confidence

    if (decision == TradingDecision::BUY) {
        double cash_to_spend = portfolio_.getCashBalance() * trade_fraction * confidence;
        quantity = cash_to_spend / current_price;
    } else if (decision == TradingDecision::SELL) {
        quantity = portfolio_.getCoinBalance() * trade_fraction * confidence;
    }
    // Ensure quantity is not excessively small or negative
    return std::max(0.0, quantity);
}

// Definition for _logDecision
void AutonomousTradingAgent::_logDecision(const DecisionRecord& record) {
    if (!log_file_.is_open()) return;

    // Convert timestamp to a readable format
    std::time_t time_now = std::chrono::system_clock::to_time_t(record.timestamp);
    std::tm tm_now = *std::localtime(&time_now); // Using localtime for local timezone

    // Use the decision_log_buffer_ for formatting if desired, or write directly
    std::stringstream temp_ss;
    temp_ss << "[Decision Record] Tick: " << record.tick_index
            << ", Time: " << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S")
            << ", Price: " << std::fixed << std::setprecision(2) << record.price_at_decision
            << ", Action: " << (record.decision == TradingDecision::BUY ? "BUY" : (record.decision == TradingDecision::SELL ? "SELL" : "HOLD"))
            << ", Confidence: " << std::fixed << std::setprecision(4) << record.confidence
            << ", Quantity: " << std::fixed << std::setprecision(6) << record.quantity
            << ", Portfolio Value Before: " << std::fixed << std::setprecision(2) << record.portfolio_value_before
            << ", Reward After: " << std::fixed << std::setprecision(4) << record.reward_after
            << std::endl;
    log_file_ << temp_ss.str();
    log_file_.flush(); // Ensure it's written immediately
}


void AutonomousTradingAgent::appendPriceTick(const PriceTick& current_tick) {
    price_series_.push_back(current_tick);
    if (price_series_.size() > max_price_series_size_) {
        price_series_.erase(price_series_.begin()); // Keep the series size bounded
    }
    tech_analyzer_.updatePriceSeries(price_series_); 
    
    // std::cout << \"[AutonomousTradingAgent] Appended tick. Series size: \" << price_series_.size() << std::endl;
    // log_file_ << \"Appended tick. Series size: \" << price_series_.size() << std::endl;
}

void AutonomousTradingAgent::setFullPriceSeries(const std::vector<PriceTick>& price_series) {
    price_series_ = price_series;
    // Ensure series does not exceed max_price_series_size_ if needed, or assume it\'s managed externally.
    if (price_series_.size() > max_price_series_size_) {
        // Option 1: Truncate
        price_series_.erase(price_series_.begin(), price_series_.begin() + (price_series_.size() - max_price_series_size_));
        // Option 2: Log warning or error, as this method implies setting the exact series provided.
    }
    tech_analyzer_.updatePriceSeries(price_series_);
    
    std::cout << "[AutonomousTradingAgent] Set full price series with " << price_series_.size() << " ticks" << std::endl;
    log_file_ << "Set full price series with " << price_series_.size() << " ticks" << std::endl;
}

// Stub implementations for saveState and loadState
bool AutonomousTradingAgent::saveState(const std::string& file_prefix) const {
    std::cout << "AutonomousTradingAgent::saveState called with prefix: " << file_prefix << std::endl;
    // TODO: Implement actual state saving logic
    // Example: Save neural network state, any learned parameters, etc.
    // For now, just return true to indicate success.
    
    // Create a dummy file to simulate saving
    std::ofstream outfile(file_prefix + "_agent_state.json");
    if (!outfile) {
        std::cerr << "Error: Could not open agent state file for writing: " << file_prefix + "_agent_state.json" << std::endl;
        return false;
    }
    nlohmann::json agent_state_json;
    agent_state_json["agent_type"] = "AutonomousTradingAgent";
    agent_state_json["last_decision_price"] = last_decision_price_;
    // Add other relevant state variables here
    
    outfile << agent_state_json.dump(4); // pretty print JSON
    outfile.close();
    
    std::cout << "AutonomousTradingAgent state placeholder saved to " << file_prefix + "_agent_state.json" << std::endl;
    return true;
}

bool AutonomousTradingAgent::loadState(const std::string& file_prefix) {
    std::cout << "AutonomousTradingAgent::loadState called with prefix: " << file_prefix << std::endl;
    // TODO: Implement actual state loading logic
    // Example: Load neural network state, any learned parameters, etc.
    // For now, just return true to indicate success.

    std::ifstream infile(file_prefix + "_agent_state.json");
    if (!infile) {
        std::cerr << "Error: Could not open agent state file for reading: " << file_prefix + "_agent_state.json" << std::endl;
        return false;
    }
    
    try {
        nlohmann::json agent_state_json;
        infile >> agent_state_json;
        infile.close();

        if (agent_state_json.contains("agent_type") && agent_state_json["agent_type"] == "AutonomousTradingAgent") {
            last_decision_price_ = agent_state_json.value("last_decision_price", 0.0);
            // Load other relevant state variables here
            std::cout << "AutonomousTradingAgent state placeholder loaded from " << file_prefix + "_agent_state.json" << std::endl;
            return true;
        } else {
            std::cerr << "Error: Agent state file does not contain valid AutonomousTradingAgent data or is missing 'agent_type'." << std::endl;
            return false;
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Error parsing agent state JSON: " << e.what() << std::endl;
        if (infile.is_open()) infile.close();
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading agent state: " << e.what() << std::endl;
        if (infile.is_open()) infile.close();
        return false;
    }
}
