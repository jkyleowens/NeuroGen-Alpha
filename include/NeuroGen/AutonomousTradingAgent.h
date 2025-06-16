#ifndef NEUROGEN_AUTONOMOUSTRADINGAGENT_H
#define NEUROGEN_AUTONOMOUSTRADINGAGENT_H

#include <string>
#include <vector>
#include <map>
#include <chrono> // For DecisionRecord timestamp
#include <fstream> // For log_file_
#include <sstream> // Required for std::stringstream
#include <nlohmann/json.hpp> // For JSON operations

// Forward declarations or includes for dependent classes
#include <NeuroGen/Portfolio.h>             // Will be copied to include/NeuroGen/
#include <NeuroGen/TechnicalAnalysis.h>   // Will be copied to include/NeuroGen/
#include <NeuroGen/NeuralNetworkInterface.h>// Will be copied to include/NeuroGen/
#include <NeuroGen/PriceTick.h>             // For PriceTick structure

class AutonomousTradingAgent {
public:
    enum class TradingDecision {
        BUY,
        SELL,
        HOLD
    };

    struct DecisionRecord {
        int tick_index; // Added
        TradingDecision decision;
        double confidence;
        double price_at_decision; // Renamed from price
        double quantity; // Quantity traded for this decision (0 if HOLD or trade failed/not executed)
        std::chrono::system_clock::time_point timestamp;
        double portfolio_value_before; // Added
        double reward_after; // Added
        // Potentially add P&L after trade, etc.
    };

    AutonomousTradingAgent(CoinbaseAdvancedTradeApi* api_client = nullptr); // Modified constructor
    ~AutonomousTradingAgent();

    /**
     * @brief Initializes the agent for a specific trading symbol.
     *        Portfolio and TechnicalAnalysis are initialized with default or empty states.
     *        NeuralNetworkInterface is initialized.
     * @param symbol The trading symbol (e.g., "BTC-USD" for Coinbase).
     * @param initial_cash The starting cash for the portfolio.
     * @param coinbase_api_ptr Pointer to the CoinbaseAdvancedTradeApi instance (can be nullptr).
     * @return True if initialization was successful, false otherwise.
     */
    bool initialize(const std::string& symbol, double initial_cash, CoinbaseAdvancedTradeApi* coinbase_api_ptr); // Modified signature

    /**
     * @brief Makes a trading decision based on the current market state.
     * @param tick_index The current index in the time series data.
     * @param current_price The current market price of the asset.
     * @return The trading decision (BUY, SELL, HOLD).
     */
    TradingDecision makeDecision(int tick_index, double current_price);

    /**
     * @brief Makes a trading decision based on the current market state using full OHLCV data.
     * This version is preferred for CSV-based simulation as it provides more accurate technical analysis.
     * @param tick_index The current index in the time series data.
     * @param price_tick The complete OHLCV data for the current tick.
     * @return The trading decision (BUY, SELL, HOLD).
     */
    TradingDecision makeDecision(int tick_index, const PriceTick& price_tick);

    /**
     * @brief Receives a reward signal from the simulation environment.
     * This reward is typically passed to the neural network interface for learning.
     * @param reward The reward value.
     */
    void receiveReward(double reward);

    /**
     * @brief Appends a new price tick to the agent's internal price series and updates TA.
     * @param current_tick The latest PriceTick data.
     */
    void appendPriceTick(const PriceTick& current_tick); // Changed from setPriceSeries

    /**
     * @brief Sets or updates the full price series for the technical analyzer and the agent's internal copy.
     * @param price_series The vector of PriceTick data.
     */
    void setFullPriceSeries(const std::vector<PriceTick>& price_series);

    /**
     * @brief Saves the agent's state (including NN state, portfolio, etc.).
     * @param filename The base filename for saving state files.
     * @return True if successful, false otherwise.
     */
    bool saveState(const std::string& filename) const;

    /**
     * @brief Loads the agent's state.
     * @param filename The base filename for loading state files.
     * @return True if successful, false otherwise.
     */
    bool loadState(const std::string& filename);

    const std::vector<DecisionRecord>& getDecisionHistory() const { return decision_history_; }
    const Portfolio& getPortfolio() const { return portfolio_; }
    double getCumulativeReward() const { return cumulative_reward_; }
    double getEpsilon() const { return epsilon_; }
    bool isInitialized() const { return is_initialized_; } // Added getter

private:
    // Helper methods for decision making and state management
    TradingDecision _determineActionFromSignal(double prediction_signal, double& confidence);
    std::map<std::string, double> _getCurrentMarketFeatures(int current_tick_index, double current_price);
    double _determineQuantity(double current_price, TradingDecision decision, double confidence);
    void _logDecision(const DecisionRecord& record); // Assuming this is also a private helper, was called in makeDecision

    // Reordered to generally match typical initialization order or logical grouping
    std::string symbol_;
    CoinbaseAdvancedTradeApi* coinbase_api_ptr_; // Pointer to the CoinbaseAdvancedTradeApi instance
    Portfolio portfolio_;
    NeuralNetworkInterface nn_interface_;
    TechnicalAnalysis tech_analyzer_; // Initialized with price_series_

    std::vector<PriceTick> price_series_;
    size_t max_price_series_size_; // Added

    std::vector<DecisionRecord> decision_history_;
    std::vector<double> reward_history_;
    double cumulative_reward_;
    double epsilon_; // For exploration-exploitation strategy
    double last_decision_price_; // Added
    static constexpr double EPSILON_DECAY = 0.995;

    bool is_initialized_; // Added
    std::ofstream log_file_;
    std::stringstream decision_log_buffer_; // Added

    // Private helper methods as in the C++ file
    TradingDecision _determineAction(double predicted_price, double current_price, double& confidence);
    double _calculateConfidence(double predicted_price, double current_price);
    double _determineQuantity(TradingDecision decision, double confidence, double current_price);
};

#endif // NEUROGEN_AUTONOMOUSTRADINGAGENT_H