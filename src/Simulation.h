#ifndef NEUROGEN_SIMULATION_H
#define NEUROGEN_SIMULATION_H

#include <string>
#include <vector>
#include <fstream>
#include "NeuroGen/PriceTick.h"
#include "NeuroGen/AutonomousTradingAgent.h"
#include "NeuroGen/CoinbaseAdvancedTradeApi.h"

class Simulation {
public:
    Simulation(CoinbaseAdvancedTradeApi* api_client);
    ~Simulation();

    /**
     * @brief Initializes the simulation with pre-fetched historical data and an API client.
     * @param symbol The trading pair symbol (e.g., "BTC-USD").
     * @param initial_cash The initial cash for the trading agent.
     * @param initial_price_data The pre-fetched historical price data.
     * @param api_client Pointer to the initialized CoinbaseAdvancedTradeApi client.
     * @return True if initialization was successful, false otherwise.
     */
    bool initialize(const std::string& symbol, double initial_cash, const std::vector<PriceTick>& initial_price_data, CoinbaseAdvancedTradeApi* api_client);

    /**
     * @brief Runs the main simulation loop.
     * Iterates through the timeSeriesData, making decisions and calculating rewards.
     * @param max_ticks Maximum number of ticks to process. If 0 or negative, processes all remaining ticks.
     */
    void run(int max_ticks = 0);

    /**
     * @brief Advances the simulation by one tick.
     * Gets current data, calls agent.makeDecision(), calculates reward, and calls agent.receiveReward().
     * @return True if a tick was advanced, false if at the end of data or not initialized.
     */
    bool advanceTick();

    // Getter methods for simulation state or results if needed
    int getCurrentTickIndex() const;
    const std::vector<PriceTick>& getTimeSeriesData() const;
    const AutonomousTradingAgent& getAgent() const; // If access to agent details is needed post-simulation

    /**
     * @brief Saves the current simulation state to files.
     * @param filename Base filename for state files.
     * @return True if state was saved successfully, false otherwise.
     */
    bool saveState(const std::string& filename);

    /**
     * @brief Loads simulation state from files.
     * @param filename Base filename for state files.
     * @return True if state was loaded successfully, false otherwise.
     */
    bool loadState(const std::string& filename);

private:
    std::vector<PriceTick> time_series_data_;
    int current_tick_index_;
    AutonomousTradingAgent agent_;
    CoinbaseAdvancedTradeApi* coinbase_api_ptr_;
    std::string trading_pair_;
    bool is_running_;
    std::ofstream simulation_log_;

    /**
     * @brief Calculates the reward for a trading decision.
     * @param decision The decision made by the agent.
     * @param price_at_decision The price when the decision was made.
     * @param price_after_decision The price at the next tick (to evaluate outcome).
     * @param quantity_traded The quantity of asset traded.
     * @return The calculated reward value.
     */
    double _calculateReward(
        AutonomousTradingAgent::TradingDecision decision,
        double price_at_decision,
        double price_after_decision,
        double quantity_traded // May need more portfolio state or trade details
    );
    
    /**
     * @brief Logs a simulation step to the CSV log file.
     * @param tick_index_at_decision The tick index when the decision was made.
     * @param price_at_decision The price when the decision was made.
     * @param portfolio_value_at_decision The portfolio value at decision time.
     * @param reward The calculated reward.
     */
    void _logSimulationStep(int tick_index_at_decision, double price_at_decision, double portfolio_value_at_decision, double reward);
};

#endif // NEUROGEN_SIMULATION_H
