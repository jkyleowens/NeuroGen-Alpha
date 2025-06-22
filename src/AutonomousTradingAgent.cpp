#include <NeuroGen/AutonomousTradingAgent.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <nlohmann/json.hpp> // Added for JSON operations
#include <sstream>      // For std::stringstream, already in .h but good practice

using namespace NeuroGen;

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

AutonomousTradingAgent::AutonomousTradingAgent(const std::string& symbol,
                                                 const NeuralNetworkInterface::Config& nn_config,
                                                 Portfolio& portfolio_ref,
                                                 CoinbaseAdvancedTradeApi* api_client)
    : symbol_(symbol),
      coinbase_api_ptr_(api_client),
      portfolio_ref_(portfolio_ref),
      nn_interface_(nn_config),
      tech_analyzer_(price_series_),
      max_price_series_size_(200),
      cumulative_reward_(0.0),
      epsilon_(1.0),
      last_decision_price_(0.0),
      is_initialized_(false)
{
    log_file_.open("agent_" + symbol_ + "_log.txt");
    if (!log_file_.is_open()) {
        std::cerr << "[AutonomousTradingAgent] Warning: Could not open log file for " << symbol_ << std::endl;
    }
    
    if (!nn_interface_.initialize()) {
        std::cerr << "[AutonomousTradingAgent] Warning: Neural Network Interface failed to initialize for " << symbol_ << std::endl;
    }
    
    price_series_.clear();
    decision_history_.clear();
    reward_history_.clear();
    is_initialized_ = true;
    
    if (coinbase_api_ptr_) {
        std::cout << "[AutonomousTradingAgent] Agent initialized for " << symbol_ << " with API client." << std::endl;
    } else {
        std::cout << "[AutonomousTradingAgent] Agent initialized for " << symbol_ << " without API client (CSV mode)." << std::endl;
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
//     portfolio_ref_.reset(initial_cash, 0.0); 

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



AutonomousTradingAgent::DecisionRecord AutonomousTradingAgent::makeDecision(int current_tick_index, const PriceTick& price_tick) {
    std::cout << "[Agent] Making decision using full OHLCV data" << std::endl;
    std::cout << "[Agent] Price data: O=" << price_tick.open << " H=" << price_tick.high 
              << " L=" << price_tick.low << " C=" << price_tick.close << " V=" << price_tick.volume << std::endl;
    
    if (!is_initialized_) {
        std::cerr << "[Agent] Error: Agent not initialized." << std::endl;
        DecisionRecord empty_record{};
        empty_record.decision = TradingDecision::HOLD;
        empty_record.tick_index = current_tick_index;
        return empty_record;
    }

    // Use the complete price tick data
    appendPriceTick(price_tick);
    
    double current_price = price_tick.close; // Use closing price for decision making
    
    std::map<std::string, double> features;
    if (nn_interface_.isInitialized()) {
        features = _getCurrentMarketFeatures(price_tick);
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
            trade_executed = portfolio_ref_.executeBuy(quantity_to_trade, current_price);
        } else if (decision == TradingDecision::SELL) {
            trade_executed = portfolio_ref_.executeSell(quantity_to_trade, current_price);
        }
    }

    if (quantity_to_trade > 0 && !trade_executed) {
        std::cout << "[Agent] Trade for " << quantity_to_trade << " units failed. Reverting to HOLD." << std::endl;
        decision = TradingDecision::HOLD;
        quantity_to_trade = 0;
    }

    DecisionRecord current_decision_record;
    current_decision_record.tick_index = current_tick_index;
    current_decision_record.timestamp = std::chrono::system_clock::now();
    current_decision_record.price_at_decision = current_price;
    current_decision_record.decision = decision;
    current_decision_record.confidence = confidence;
    current_decision_record.quantity = quantity_to_trade;
    current_decision_record.portfolio_value_before = portfolio_ref_.getCurrentValue(current_price);
    current_decision_record.reward_after = 0.0;

    decision_history_.push_back(current_decision_record);
    _logDecision(current_decision_record);

    if (log_file_.is_open()) {
        log_file_ << "[Full OHLCV Decision] Tick: " << current_tick_index 
                  << ", OHLCV: " << price_tick.open << "/" << price_tick.high << "/" << price_tick.low << "/" << price_tick.close << "/" << price_tick.volume
                  << ", Action: " << (decision == TradingDecision::BUY ? "BUY" : (decision == TradingDecision::SELL ? "SELL" : "HOLD"))
                  << ", Quantity: " << quantity_to_trade
                  << ", Portfolio Value: " << portfolio_ref_.getCurrentValue(current_price)
                  << ", Cash: " << portfolio_ref_.getCashBalance()
                  << ", Coins: " << portfolio_ref_.getCoinBalance()
                  << ", Epsilon: " << epsilon_ << std::endl;
    }

    return current_decision_record;
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
std::map<std::string, double> AutonomousTradingAgent::_getCurrentMarketFeatures(const PriceTick& price_tick) {
    std::map<std::string, double> features;
    
    // Basic price features from the provided PriceTick
    features["current_price"] = price_tick.close;

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
    if (!price_series_.empty()) {
        // Use the last available index in the price series for TA calculations
        int ta_index = static_cast<int>(price_series_.size()) - 1;
        
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
                features["bb_position"] = (price_tick.close - ta_features["BB_Lower_20"]) / bb_range;
            }
        }

    } else {
        std::cout << "[Agent_Features] Price series is empty, cannot calculate TA features." << std::endl;
    }

    // Portfolio-related features
    features["cash_balance"] = portfolio_ref_.getCashBalance();
    features["coin_balance"] = portfolio_ref_.getCoinBalance();
    features["portfolio_value"] = portfolio_ref_.getCurrentValue(price_tick.close);
    
    // Normalized portfolio allocation
    double total_value = portfolio_ref_.getCurrentValue(price_tick.close);
    if (total_value > 0) {
        features["cash_ratio"] = portfolio_ref_.getCashBalance() / total_value;
        features["coin_ratio"] = (portfolio_ref_.getCoinBalance() * price_tick.close) / total_value;
    }

    std::cout << "[Agent_Features] Generated " << features.size() << " features for the latest tick." << std::endl;
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
        double cash_to_spend = portfolio_ref_.getCashBalance() * trade_fraction * confidence;
        quantity = cash_to_spend / current_price;
    } else if (decision == TradingDecision::SELL) {
        quantity = portfolio_ref_.getCoinBalance() * trade_fraction * confidence;
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

// Implementations for saveState and loadState
bool AutonomousTradingAgent::saveState(const std::string& file_prefix) const {
    std::cout << "AutonomousTradingAgent::saveState called with prefix: " << file_prefix << std::endl;
    
    // Save the main agent state
    nlohmann::json agent_state_json;
    agent_state_json["agent_type"] = "AutonomousTradingAgent";
    agent_state_json["symbol"] = symbol_;
    agent_state_json["epsilon"] = epsilon_;
    agent_state_json["cumulative_reward"] = cumulative_reward_;
    agent_state_json["last_decision_price"] = last_decision_price_;

    std::ofstream agent_outfile(file_prefix + "_agent_state.json");
    if (!agent_outfile) {
        std::cerr << "Error: Could not open agent state file for writing: " << file_prefix + "_agent_state.json" << std::endl;
        return false;
    }
    agent_outfile << agent_state_json.dump(4); // pretty print JSON
    agent_outfile.close();

    // Save the neural network state
    if (!nn_interface_.saveState(file_prefix + "_nn_state.json")) {
        std::cerr << "Error: Failed to save neural network state." << std::endl;
        return false;
    }

    // Save the portfolio state
    if (!portfolio_ref_.saveState(file_prefix + "_portfolio_state.json")) {
        std::cerr << "Error: Failed to save portfolio state." << std::endl;
        return false;
    }
    
    std::cout << "AutonomousTradingAgent full state saved with prefix: " << file_prefix << std::endl;
    return true;
}

bool AutonomousTradingAgent::loadState(const std::string& file_prefix) {
    std::cout << "AutonomousTradingAgent::loadState called with prefix: " << file_prefix << std::endl;
    
    // Load the main agent state
    std::ifstream agent_infile(file_prefix + "_agent_state.json");
    if (!agent_infile) {
        std::cerr << "Error: Could not open agent state file for reading: " << file_prefix + "_agent_state.json" << std::endl;
        return false;
    }
    
    nlohmann::json agent_state_json;
    try {
        agent_infile >> agent_state_json;
        agent_infile.close();

        if (!agent_state_json.contains("agent_type") || agent_state_json["agent_type"] != "AutonomousTradingAgent") {
            std::cerr << "Error: Agent state file is not for AutonomousTradingAgent." << std::endl;
            return false;
        }
        
        symbol_ = agent_state_json.value("symbol", symbol_);
        epsilon_ = agent_state_json.value("epsilon", epsilon_);
        cumulative_reward_ = agent_state_json.value("cumulative_reward", 0.0);
        last_decision_price_ = agent_state_json.value("last_decision_price", 0.0);

    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Error parsing agent state JSON: " << e.what() << std::endl;
        if(agent_infile.is_open()) agent_infile.close();
        return false;
    }

    // Load the neural network state
    if (!nn_interface_.loadState(file_prefix + "_nn_state.json")) {
        std::cerr << "Error: Failed to load neural network state." << std::endl;
        return false;
    }

    // Load the portfolio state
    if (!portfolio_ref_.loadState(file_prefix + "_portfolio_state.json")) {
        std::cerr << "Error: Failed to load portfolio state." << std::endl;
        return false;
    }
    
    std::cout << "AutonomousTradingAgent full state loaded from prefix: " << file_prefix << std::endl;
    is_initialized_ = true;
    return true;
}