#include <NeuroGen/Simulation.h>
#include <NeuroGen/AutonomousTradingAgent.h>
#include <NeuroGen/CoinbaseAdvancedTradeApi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <sstream> 
#include <algorithm>

Simulation::Simulation(CoinbaseAdvancedTradeApi* api_client)
    : current_tick_index_(0), is_running_(false), coinbase_api_ptr_(api_client), agent_(api_client) {
    
    // Open simulation log file
    simulation_log_.open("simulation_log.csv");
    if (simulation_log_.is_open()) {
        // Write header
        simulation_log_ << "timestamp,tick_index,price,portfolio_value,reward,action,confidence,quantity_traded,cash_balance,coin_balance\\\\n";
    } else {
        std::cerr << "[Simulation] Warning: Could not open simulation log file" << std::endl;
    }
    
    if (api_client) {
        std::cout << "[Simulation] Simulation initialized with API client." << std::endl;
    } else {
        std::cout << "[Simulation] Simulation initialized without API client (CSV mode)." << std::endl;
    }
}

Simulation::~Simulation() {
    if (simulation_log_.is_open()) {
        simulation_log_.close();
    }
    
    std::cout << "[Simulation] Simulation destroyed" << std::endl;
}

bool Simulation::initialize(const std::string& symbol, double initial_cash, const std::vector<PriceTick>& initial_price_data, CoinbaseAdvancedTradeApi* api_client) {
    trading_pair_ = symbol;
    coinbase_api_ptr_ = api_client; // Store the API client pointer

    // It's okay if coinbase_api_ptr_ is null (e.g., when running from CSV)
    // The agent will need to handle a null API client.
    // if (!coinbase_api_ptr_) { // Removed this check
    //     std::cerr << "[Simulation] Error: Coinbase API client is null." << std::endl;
    //     return false;
    // }

    time_series_data_ = initial_price_data;

    if (time_series_data_.empty()) {
        std::cerr << "[Simulation] Error: No initial price data provided for " << trading_pair_ << std::endl;
        // It might be valid to initialize with no data if it's loaded later via loadState,
        // but for a fresh initialize, data is expected.
        // However, main.cpp already loads data before calling this.
    }

    // Initialize the agent. Pass the API client pointer.
    // The agent's initialize method will be updated to accept a pointer.
    if (!agent_.initialize(trading_pair_, initial_cash, coinbase_api_ptr_)) { 
        std::cerr << "[Simulation] Error: Failed to initialize AutonomousTradingAgent." << std::endl;
        return false;
    }
    agent_.setFullPriceSeries(time_series_data_); // Agent needs the price series for TA etc.

    current_tick_index_ = 0;
    is_running_ = false; // Will be set to true when run() is called
    std::cout << "[Simulation] Initialized with " << time_series_data_.size() << " data points for " << trading_pair_ << std::endl;
    if (coinbase_api_ptr_) {
        std::cout << "[Simulation] API client is active." << std::endl;
    } else {
        std::cout << "[Simulation] API client is NULL (CSV/offline mode)." << std::endl;
    }
    return true;
}

void Simulation::run(int max_ticks) {
    if (time_series_data_.empty()) {
        std::cerr << "[Simulation] Error: No historical data available. Initialize simulation first." << std::endl;
        return;
    }
    if (current_tick_index_ >= static_cast<int>(time_series_data_.size())) {
        std::cout << "[Simulation] Already at the end of data. Reset or re-initialize." << std::endl;
        return;
    }
    
    is_running_ = true;
    std::cout << "[Simulation] Starting simulation..." << std::endl;
    
    int ticks_to_process = (max_ticks > 0) 
                           ? std::min(max_ticks, static_cast<int>(time_series_data_.size() - current_tick_index_)) 
                           : static_cast<int>(time_series_data_.size() - current_tick_index_);
    
    for (int i = 0; i < ticks_to_process && is_running_; ++i) {
        if (!advanceTick()) { // advanceTick now returns bool
            break; // Stop if advanceTick indicates an issue or end of data
        }
        // Optional: std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    is_running_ = false; // Ensure is_running_ is false after loop, even if break occurs
    
    if (current_tick_index_ > 0 && current_tick_index_ <= static_cast<int>(time_series_data_.size())) {
        double final_price = time_series_data_[current_tick_index_ -1].close; // Price at the last processed tick
        const Portfolio& portfolio = agent_.getPortfolio(); // Use const reference

        std::cout << "\\n[Simulation] Simulation completed" << std::endl;
        std::cout << "Ticks processed: " << (current_tick_index_ - (static_cast<int>(time_series_data_.size()) - ticks_to_process)) << std::endl;
        std::cout << "Current tick index: " << current_tick_index_ << std::endl;
        std::cout << "Final portfolio value: $" << portfolio.getCurrentValue(final_price) << std::endl;
        // ROI calculation:
        // double initial_portfolio_value = portfolio.getInitialValue(); // Assuming Portfolio stores this
        // if (initial_portfolio_value > 0) {
        //     double roi = ((portfolio.getCurrentValue(final_price) - initial_portfolio_value) / initial_portfolio_value) * 100.0;
        //     std::cout << "Return on investment: " << roi << "%" << std::endl;
        // }
        std::cout << "Cash balance: $" << portfolio.getCashBalance() << std::endl;
        std::cout << "Coin balance: " << portfolio.getCoinBalance() << " " << trading_pair_ << std::endl;
    } else if (time_series_data_.empty()) {
        std::cout << "\\n[Simulation] Simulation ended (no data)." << std::endl;
    } else {
        std::cout << "\\n[Simulation] Simulation ended." << std::endl;
         const Portfolio& portfolio = agent_.getPortfolio();
         // Attempt to log final state even if loop didn't run due to pre-conditions
         double last_known_price = !time_series_data_.empty() ? time_series_data_[0].close : 0.0;
         if (current_tick_index_ > 0 && current_tick_index_ <= static_cast<int>(time_series_data_.size())) {
             last_known_price = time_series_data_[current_tick_index_-1].close;
         } else if (!time_series_data_.empty() && current_tick_index_ == 0) {
            // If simulation was initialized but not run, use first tick's price for portfolio value
            last_known_price = time_series_data_[0].close;
         }

        std::cout << "Final portfolio value: $" << portfolio.getCurrentValue(last_known_price) << std::endl;
        std::cout << "Cash balance: $" << portfolio.getCashBalance() << std::endl;
        std::cout << "Coin balance: " << portfolio.getCoinBalance() << " " << trading_pair_ << std::endl;
    }
}

bool Simulation::advanceTick() {
    if (current_tick_index_ >= static_cast<int>(time_series_data_.size())) {
        std::cout << "[Simulation] End of data reached" << std::endl;
        is_running_ = false;
        return false;
    }
    
    const PriceTick& current_tick_data = time_series_data_[current_tick_index_];
    double current_price = current_tick_data.close;
    
    std::cout << "\\n[Simulation] Processing tick " << current_tick_index_ 
              << " - Timestamp: " << current_tick_data.timestamp
              << " - OHLCV: " << current_tick_data.open << "/" << current_tick_data.high 
              << "/" << current_tick_data.low << "/" << current_tick_data.close 
              << "/" << current_tick_data.volume << std::endl;
    
    // Agent makes a decision using the full OHLCV data for better analysis
    agent_.makeDecision(current_tick_index_, current_tick_data); 
    
    // Retrieve details of the decision made by the agent for reward calculation and logging.
    // Assuming AutonomousTradingAgent stores its last decision details.
    AutonomousTradingAgent::TradingDecision decision_type = AutonomousTradingAgent::TradingDecision::HOLD;
    double quantity_traded = 0.0;
    // Access last decision details from agent (e.g., via getDecisionHistory or specific getters)
    if (!agent_.getDecisionHistory().empty()) {
        const auto& last_decision_record = agent_.getDecisionHistory().back();
        decision_type = last_decision_record.decision;
        quantity_traded = last_decision_record.quantity; // Assuming DecisionRecord has 'quantity'
    } else {
        std::cerr << "[Simulation] Warning: Agent decision history is empty after makeDecision call." << std::endl;
    }

    current_tick_index_++; // Advance to the next tick index *before* using it for next_price
    
    double reward = 0.0;
    if (current_tick_index_ < static_cast<int>(time_series_data_.size())) {
        const PriceTick& next_tick_data = time_series_data_[current_tick_index_];
        double next_price = next_tick_data.close;
        
        reward = _calculateReward(decision_type, current_price, next_price, quantity_traded);
        agent_.receiveReward(reward);
        
        _logSimulationStep(current_tick_index_ -1, current_price, // Log with price at decision
                           agent_.getPortfolio().getCurrentValue(current_price), reward); 
    } else {
        // Last tick, no next_price to calculate reward based on future.
        // Agent might still get a reward based on final portfolio state or a terminal reward.
        // For now, no reward is sent for the very last action if there's no subsequent tick.
        std::cout << "[Simulation] At final tick, no future price for reward calculation." << std::endl;
        _logSimulationStep(current_tick_index_ -1, current_price,
                           agent_.getPortfolio().getCurrentValue(current_price), 0.0); // Log with 0 reward
        is_running_ = false; // End of data
        return false; // Indicate end of simulation or inability to advance further
    }
    return true; // Tick advanced successfully
}

double Simulation::_calculateReward(AutonomousTradingAgent::TradingDecision decision, 
                                   double price_at_decision, 
                                   double price_after_decision, 
                                   double quantity_traded) {
    double price_change_pct = (price_after_decision - price_at_decision) / price_at_decision;
    double reward = 0.0;
    
    // If a trade occurred (quantity_traded > 0), the reward is more significant.
    // If it's a HOLD, the reward/penalty is for the decision to hold.
    // double trade_impact_scale = (quantity_traded > 0) ? 1.0 : 0.5; // Example: less penalty/reward for just holding
    // Currently not used but could be integrated to scale rewards by trade size

    switch (decision) {
        case AutonomousTradingAgent::TradingDecision::BUY:
            // Reward for buying if price increased, penalize if price decreased.
            reward = price_change_pct; // Positive if price_after > price_at_decision
            break;
            
        case AutonomousTradingAgent::TradingDecision::SELL:
            // Reward for selling if price decreased, penalize if price increased.
            reward = -price_change_pct; // Positive if price_after < price_at_decision
            break;
            
        case AutonomousTradingAgent::TradingDecision::HOLD:
            // Small reward for holding during stable prices or correctly avoiding bad trades.
            // Small penalty for holding during significant missed opportunities.
            if (std::abs(price_change_pct) < 0.001) { // Less than 0.1% change, considered stable
                reward = 0.01; // Small positive reward for holding during stability
            } else {
                // Penalize for holding when there was a significant move (missed opportunity)
                // The sign of price_change_pct indicates the direction of missed opportunity.
                // e.g. if price went up and agent held, it's a missed buy. If price went down, missed sell.
                // A simple penalty for volatility when holding:
                reward = -std::abs(price_change_pct) * 0.1; // Scaled penalty
            }
            break;
    }
    
    // Scale reward by trade_impact_scale (not used above, could be integrated if desired)
    // For example: reward *= trade_impact_scale;
    // Or, if quantity_traded is substantial, it could amplify the reward/penalty.
    // E.g., if (quantity_traded > 0) reward *= (1 + log10(std::max(1.0, quantity_traded * price_at_decision / 1000.0))); // Scale by order value magnitude

    // Clamp reward to a standard range, e.g., [-1, 1]
    reward = std::max(-1.0, std::min(1.0, reward));
    
    std::cout << "[Simulation] Reward: " << reward 
              << " (Decision: " << static_cast<int>(decision)
              << ", Qty: " << quantity_traded
              << ", PriceChange: " << (price_change_pct * 100.0) << "%)" << std::endl;
    
    return reward;
}

void Simulation::_logSimulationStep(int tick_index_at_decision, double price_at_decision, double portfolio_value_at_decision, double reward) {
    if (!simulation_log_.is_open()) {
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    // Use std::gmtime for UTC or std::localtime for local time.
    // For consistency in logs, UTC is often preferred if timestamps in data are UTC.
    std::tm tm_snapshot = *std::localtime(&time_t_now); // Or *std::gmtime(&time_t_now)
    
    int action_taken_int = 0; // Default to HOLD
    double confidence = 0.0;
    double quantity_traded_log = 0.0;

    if (!agent_.getDecisionHistory().empty()) {
        const auto& last_decision_record = agent_.getDecisionHistory().back();
        action_taken_int = static_cast<int>(last_decision_record.decision);
        confidence = last_decision_record.confidence;
        quantity_traded_log = last_decision_record.quantity;
    }
    
    simulation_log_ << std::put_time(&tm_snapshot, "%Y-%m-%d %H:%M:%S") << ","
                   << tick_index_at_decision << "," // Log tick index when decision was made
                   << price_at_decision << ","    // Log price when decision was made
                   << portfolio_value_at_decision << ","
                   << reward << ","
                   << action_taken_int << ","
                   << confidence << ","
                   << quantity_traded_log << ","
                   << agent_.getPortfolio().getCashBalance() << ","
                   << agent_.getPortfolio().getCoinBalance() << std::endl; // Log coin balance using trading_pair_
    
    simulation_log_.flush();
}

bool Simulation::saveState(const std::string& filename) {
    // Agent saves its own state, which should include its API configuration or be able to re-link.
    bool agent_saved = agent_.saveState(filename + "_agent_state.dat"); 
    
    std::ofstream state_file(filename + "_simulation_state.dat", std::ios::binary | std::ios::trunc);
    if (!state_file.is_open()) {
        std::cerr << "[Simulation] Error: Could not open state file for writing: " << filename + "_simulation_state.dat" << std::endl;
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
    
    state_file.close();
    std::cout << "[Simulation] Simulation state saved to " << filename << "_simulation_state.dat" << std::endl;
    return agent_saved && state_file.good();
}

bool Simulation::loadState(const std::string& filename) {
    // Agent loads its own state. It should handle its API client pointer internally or via its initialize method.
    // The Simulation class provides the API client pointer during its own initialization.
    bool agent_loaded = agent_.loadState(filename + "_agent_state.dat"); 
    
    std::ifstream state_file(filename + "_simulation_state.dat", std::ios::binary);
    if (!state_file.is_open()) {
        std::cerr << "[Simulation] Warning: No simulation state file found: " << filename + "_simulation_state.dat" << "." << std::endl;
        // If agent loaded but sim state is missing, this implies that the simulation needs to be initialized
        // with new data, but the agent's learned parameters are preserved.
        // The caller (main.cpp) handles fetching new data and calling simulation.initialize().
        // So, we just return agent_loaded status. The simulation's time_series_data_ will be empty.
        current_tick_index_ = 0;
        time_series_data_.clear();
        // trading_pair_ will be set by the subsequent call to initialize() in main.cpp
        return agent_loaded; 
    }
    
    try {
        state_file.read(reinterpret_cast<char*>(&current_tick_index_), sizeof(current_tick_index_));
        if(state_file.gcount() != sizeof(current_tick_index_)) throw std::runtime_error("Failed to read current_tick_index_ or EOF reached");

        size_t data_size;
        state_file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if(state_file.gcount() != sizeof(data_size)) throw std::runtime_error("Failed to read data_size or EOF reached");
        
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
            if(state_file.gcount() != sizeof(PriceTick)) throw std::runtime_error("Failed to read a full PriceTick or EOF reached");
            time_series_data_.push_back(tick);
        }
        
        if (!time_series_data_.empty()) {
             // Ensure agent has the loaded series and the correct API client.
             // The API client is already set in the agent via the Simulation constructor.
             // If agent_.initialize is called here, it would reset the agent. 
             // We assume agent_.loadState() correctly restores the agent, and it uses the API client from its constructor.
             agent_.setFullPriceSeries(time_series_data_); 
        } else if (data_size > 0) {
            // This case (data_size > 0 but time_series_data_ is empty) indicates a problem during PriceTick reading loop.
             throw std::runtime_error("Data size mismatch after reading PriceTicks.");
        }
        
        state_file.close();
        std::cout << "[Simulation] Simulation state loaded from " << filename << "_simulation_state.dat" << std::endl;
        std::cout << "[Simulation] Resuming at tick index: " << current_tick_index_ 
                  << ", Data size: " << time_series_data_.size() << std::endl;
        
        return agent_loaded && state_file.good();
    } catch (const std::exception& e) {
        std::cerr << "[Simulation] Error loading simulation state: " << e.what() << std::endl;
        // Reset to a clean state if loading fails mid-way
        current_tick_index_ = 0;
        time_series_data_.clear();
        is_running_ = false;
        return false;
    }
}
