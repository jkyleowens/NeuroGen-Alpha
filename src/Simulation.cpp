#include "NeuroGen/Simulation.h"
#include "NeuroGen/AutonomousTradingAgent.h"
#include "NeuroGen/Portfolio.h" // Correct include path

#include <iostream>
#include <fstream>
#include <iomanip> 
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace NeuroGen {

Simulation::Simulation(
    AutonomousTradingAgent& agent,
    Portfolio& portfolio,
    const std::vector<PriceTick>& initial_price_data,
    CoinbaseAdvancedTradeApi* api_client)
    : agent_(agent),
      portfolio_(portfolio),
      time_series_data_(initial_price_data),
      current_tick_index_(0),
      coinbase_api_ptr_(api_client),
      is_running_(false) {

    simulation_log_.open("simulation_log.csv", std::ios_base::app);
    if (simulation_log_.is_open()) {
        simulation_log_.seekp(0, std::ios::end);
        if (simulation_log_.tellp() == 0) {
            simulation_log_ << "log_timestamp,epoch_tick_index,data_timestamp,price_open,price_high,price_low,price_close,price_volume,"
                            << "portfolio_cash,portfolio_asset_value,portfolio_total_value,reward,action,confidence,quantity_traded\n";
        }
    } else {
        std::cerr << "[Simulation] Warning: Could not open simulation log file." << std::endl;
    }

    if (!time_series_data_.empty()) {
        std::cout << "[Simulation] Initialized with " << time_series_data_.size() << " data points." << std::endl;
    } else {
        std::cout << "[Simulation] Warning: Initialized with no price data." << std::endl;
    }
    
    if (coinbase_api_ptr_) {
        std::cout << "[Simulation] API client is active." << std::endl;
    } else {
        std::cout << "[Simulation] API client is NULL (offline/CSV mode)." << std::endl;
    }
}

Simulation::~Simulation() {
    if (simulation_log_.is_open()) {
        simulation_log_.close();
    }
    std::cout << "[Simulation] Simulation destroyed." << std::endl;
}

void Simulation::run(int max_ticks) {
    if (isFinished()) {
        std::cerr << "[Simulation] Error: No data or already at the end of data. Cannot run." << std::endl;
        return;
    }
    
    is_running_ = true;
    std::cout << "[Simulation] Starting simulation run..." << std::endl;
    
    int ticks_processed_in_run = 0;
    while (!isFinished() && is_running_) {
        if (max_ticks > 0 && ticks_processed_in_run >= max_ticks) {
            std::cout << "[Simulation] Max ticks for this run (" << max_ticks << ") reached." << std::endl;
            break;
        }
        if (!advanceTick()) {
            break; 
        }
        ticks_processed_in_run++;
    }
    
    is_running_ = false; 
    std::cout << "[Simulation] Simulation run finished. Ticks processed in this run: " << ticks_processed_in_run << std::endl;
}

bool Simulation::isFinished() const {
    return current_tick_index_ >= static_cast<int>(time_series_data_.size());
}

bool Simulation::advanceTick() {
    if (isFinished()) {
        is_running_ = false;
        return false;
    }
    
    const PriceTick& current_tick_data = time_series_data_[current_tick_index_];
    double portfolio_value_before_trade = portfolio_.getCurrentValue(current_tick_data.close); 

    AutonomousTradingAgent::DecisionRecord decision_details = agent_.makeDecision(current_tick_index_, current_tick_data);
    
    double portfolio_value_after_trade = portfolio_.getCurrentValue(current_tick_data.close);

    double reward = 0.0;
    const PriceTick* next_tick_data_ptr = nullptr;
    if (current_tick_index_ + 1 < static_cast<int>(time_series_data_.size())) {
        next_tick_data_ptr = &time_series_data_[current_tick_index_ + 1];
    }

    reward = _calculateReward(
        decision_details, 
        current_tick_data, 
        next_tick_data_ptr,
        portfolio_value_before_trade,
        portfolio_value_after_trade
    );
    agent_.receiveReward(reward);
    
    _logSimulationStep(
        current_tick_index_, 
        current_tick_data,
        portfolio_value_after_trade,
        reward,
        decision_details
    ); 

    current_tick_index_++;
    
    if (isFinished()) {
        is_running_ = false;
    }
    return true; 
}

double Simulation::_calculateReward(
    const AutonomousTradingAgent::DecisionRecord& decision_details, 
    const PriceTick& tick_at_decision,
    const PriceTick* tick_after_decision, 
    double portfolio_value_before_trade,
    double portfolio_value_after_trade) {

    // If there's no next tick, we can't evaluate the outcome of the decision.
    if (!tick_after_decision) {
        return 0.0;
    }

    double reward = 0.0;
    const double price_at_decision_close = tick_at_decision.close;
    const double price_after_decision_close = tick_after_decision->close;
    const double price_change_pct = (price_at_decision_close > 0) 
        ? ((price_after_decision_close - price_at_decision_close) / price_at_decision_close)
        : 0.0;

    // Define reward/penalty constants for clarity
    const double LARGE_REWARD_BONUS = 0.5;
    const double LARGE_PENALTY_FACTOR = -0.5;
    const double SMALL_REWARD = 0.1;
    const double SMALL_PENALTY = -0.15; // Make penalty for inaction slightly harsher
    const double VOLATILITY_THRESHOLD = 0.02; // Threshold to define a "significant" price move (2%)

    switch (decision_details.decision) {
        case AutonomousTradingAgent::TradingDecision::BUY:
            // If we bought and the price went up, that's a good decision.
            // The reward is proportional to how much the price increased.
            if (price_change_pct > 0) {
                reward = LARGE_REWARD_BONUS * (1 + std::min(price_change_pct * 10, 1.0)); // Reward profitable buys
            } 
            // If we bought and the price went down, that's a bad decision.
            // The penalty is proportional to how much the price dropped.
            else {
                reward = LARGE_PENALTY_FACTOR * (1 + std::min(std::abs(price_change_pct) * 10, 1.0)); // Penalize unprofitable buys
            }
            break;

        case AutonomousTradingAgent::TradingDecision::SELL:
            // If we sold and the price went down, we correctly avoided a loss.
            // The reward is proportional to the loss avoided.
            if (price_change_pct < 0) {
                reward = LARGE_REWARD_BONUS * (1 + std::min(std::abs(price_change_pct) * 10, 1.0)); // Reward profitable sells
            } 
            // If we sold and the price went up, we missed out on profit.
            // The penalty is proportional to the profit we missed.
            else {
                reward = LARGE_PENALTY_FACTOR * (1 + std::min(price_change_pct * 10, 1.0)); // Penalize missed opportunity
            }
            break;

        case AutonomousTradingAgent::TradingDecision::HOLD:
            // If we held and the market was volatile, we missed an opportunity.
            if (std::abs(price_change_pct) > VOLATILITY_THRESHOLD) {
                reward = SMALL_PENALTY;
            } 
            // If we held and the market was stable, it was a good decision to wait.
            else {
                reward = SMALL_REWARD;
            }
            break;
    }

    // Also factor in the actual change in portfolio value (PnL) as a secondary reward signal.
    double pnl_reward = 0.0;
    if (portfolio_value_before_trade > 0) {
        pnl_reward = (portfolio_value_after_trade - portfolio_value_before_trade) / portfolio_value_before_trade;
    }

    // Combine the decision-based reward and the PnL reward.
    // We can weight them, for instance, 70% for the decision quality and 30% for the raw PnL.
    double final_reward = 0.7 * reward + 0.3 * pnl_reward;
              
    // Clip the final reward to the standard range of [-1.0, 1.0] to maintain stability.
    return std::max(-1.0, std::min(1.0, final_reward));
}

void Simulation::_logSimulationStep(
    int tick_index, 
    const PriceTick& current_tick_data,
    double portfolio_value, 
    double reward,
    const AutonomousTradingAgent::DecisionRecord& decision_details) {

    if (!simulation_log_.is_open()) {
        return;
    }
    
    auto now_log = std::chrono::system_clock::now();
    auto time_t_now_log = std::chrono::system_clock::to_time_t(now_log);
    std::tm tm_snapshot_log = *std::localtime(&time_t_now_log);
    
    simulation_log_ << std::put_time(&tm_snapshot_log, "%Y-%m-%d %H:%M:%S") << ","
                   << tick_index << ","
                   << current_tick_data.timestamp << ","
                   << current_tick_data.open << ","
                   << current_tick_data.high << ","
                   << current_tick_data.low << ","
                   << current_tick_data.close << ","
                   << current_tick_data.volume << ","
                   << portfolio_.getCashBalance() << ","
                   << (portfolio_.getCoinBalance() * current_tick_data.close) << "," 
                   << portfolio_value << "," 
                   << reward << ","
                   << static_cast<int>(decision_details.decision) << ","
                   << decision_details.confidence << ","
                   << decision_details.quantity;

    simulation_log_ << std::endl;
}

int Simulation::getCurrentTickIndex() const {
    return current_tick_index_;
}

const std::vector<PriceTick>& Simulation::getTimeSeriesData() const {
    return time_series_data_;
}

const AutonomousTradingAgent& Simulation::getAgent() const {
    return agent_;
}

const Portfolio& Simulation::getPortfolio() const {
    return portfolio_;
}

bool Simulation::saveState(const std::string& filename_base) {
    std::ofstream state_file(filename_base + "_simulation_state.dat", std::ios::binary | std::ios::trunc);
    if (!state_file.is_open()) {
        std::cerr << "[Simulation] Error: Could not open state file for writing: " 
                  << filename_base + "_simulation_state.dat" << std::endl;
        return false;
    }
    
    state_file.write(reinterpret_cast<const char*>(&current_tick_index_), sizeof(current_tick_index_));
    
    size_t data_size = time_series_data_.size();
    state_file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    
    for (const auto& tick : time_series_data_) {
        state_file.write(reinterpret_cast<const char*>(&tick.timestamp), sizeof(tick.timestamp));
        state_file.write(reinterpret_cast<const char*>(&tick.open), sizeof(tick.open));
        state_file.write(reinterpret_cast<const char*>(&tick.high), sizeof(tick.high));
        state_file.write(reinterpret_cast<const char*>(&tick.low), sizeof(tick.low));
        state_file.write(reinterpret_cast<const char*>(&tick.close), sizeof(tick.close));
        state_file.write(reinterpret_cast<const char*>(&tick.volume), sizeof(tick.volume));
    }
    
    bool success = state_file.good();
    state_file.close();
    if (success) {
        std::cout << "[Simulation] Simulation state saved to " << filename_base + "_simulation_state.dat" << std::endl;
    } else {
        std::cerr << "[Simulation] Error writing simulation state to " << filename_base + "_simulation_state.dat" << std::endl;
    }
    return success;
}

bool Simulation::loadState(const std::string& filename_base) {
    std::ifstream state_file(filename_base + "_simulation_state.dat", std::ios::binary);
    if (!state_file.is_open()) {
        std::cerr << "[Simulation] Warning: No simulation state file found: " 
                  << filename_base + "_simulation_state.dat" << ". Starting fresh." << std::endl;
        current_tick_index_ = 0;
        time_series_data_.clear();
        return false;
    }
    
    try {
        state_file.read(reinterpret_cast<char*>(&current_tick_index_), sizeof(current_tick_index_));
        if(state_file.gcount() != sizeof(current_tick_index_)) throw std::runtime_error("Failed to read current_tick_index_");

        size_t data_size;
        state_file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if(state_file.gcount() != sizeof(data_size)) throw std::runtime_error("Failed to read data_size");
        
        time_series_data_.clear();
        time_series_data_.reserve(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            PriceTick tick;
            state_file.read(reinterpret_cast<char*>(&tick.timestamp), sizeof(tick.timestamp));
            state_file.read(reinterpret_cast<char*>(&tick.open), sizeof(tick.open));
            state_file.read(reinterpret_cast<char*>(&tick.high), sizeof(tick.high));
            state_file.read(reinterpret_cast<char*>(&tick.low), sizeof(tick.low));
            state_file.read(reinterpret_cast<char*>(&tick.close), sizeof(tick.close));
            state_file.read(reinterpret_cast<char*>(&tick.volume), sizeof(tick.volume));
            
            if(!state_file.good()) throw std::runtime_error("Error reading PriceTick data");
            time_series_data_.push_back(tick);
        }
        
        state_file.close();
        std::cout << "[Simulation] Simulation state loaded from " << filename_base + "_simulation_state.dat" << std::endl;
        std::cout << "[Simulation] Resuming at tick index: " << current_tick_index_ 
                  << ", Data size: " << time_series_data_.size() << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Simulation] Error loading simulation state: " << e.what() << std::endl;
        current_tick_index_ = 0;
        time_series_data_.clear();
        is_running_ = false;
        return false;
    }
}

} // namespace NeuroGen