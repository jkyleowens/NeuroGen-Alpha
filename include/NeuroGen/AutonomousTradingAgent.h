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

// Forward declare CoinbaseAdvancedTradeApi to avoid full include if only pointer is used
class CoinbaseAdvancedTradeApi; 

namespace NeuroGen { // Added namespace to match Portfolio.h and others
// Using NeuroGen namespace for consistency
class Portfolio; // Forward declaration, already included above but good practice inside namespace

class AutonomousTradingAgent {
public:
    enum class TradingDecision {
        BUY,
        SELL,
        HOLD
    };

    struct DecisionRecord {
        int tick_index; 
        TradingDecision decision;
        double confidence;
        double price_at_decision; 
        double quantity; 
        std::chrono::system_clock::time_point timestamp;
        double portfolio_value_before; 
        double reward_after; 
        // Potentially add P&L after trade, etc.
    };

    // Modified constructor: Takes symbol, NN config, portfolio reference, and optional API client
    AutonomousTradingAgent(const std::string& symbol, 
                           const NeuralNetworkInterface::Config& nn_config, // Assuming NeuralNetworkInterface::Config is defined
                           Portfolio& portfolio_ref, 
                           CoinbaseAdvancedTradeApi* api_client = nullptr);
    ~AutonomousTradingAgent();

    /**
     * @brief Makes a trading decision based on the current market state using full OHLCV data.
     * This version is preferred for CSV-based simulation as it provides more accurate technical analysis.
     * @param tick_index The current index in the time series data.
     * @param price_tick The complete OHLCV data for the current tick.
     * @return The DecisionRecord struct containing details of the decision made.
     */
    DecisionRecord makeDecision(int tick_index, const PriceTick& price_tick);

    /**
     * @brief Receives a reward signal from the simulation environment.
     * This reward is typically passed to the neural network interface for learning.
     * @param reward The reward value.
     */
    void receiveReward(double reward);

    /**
     * @brief Appends a new price tick to the agent\'s internal price series and updates TA.
     * @param current_tick The latest PriceTick data.
     */
    void appendPriceTick(const PriceTick& current_tick); 

    /**
     * @brief Sets or updates the full price series for the technical analyzer and the agent\'s internal copy.
     * @param price_series The vector of PriceTick data.
     */
    void setFullPriceSeries(const std::vector<PriceTick>& price_series);

    /**
     * @brief Saves the agent\'s state (including NN state, portfolio, etc.).
     * @param filename The base filename for saving state files.
     * @return True if successful, false otherwise.
     */
    bool saveState(const std::string& filename) const;

    /**
     * @brief Loads the agent\'s state.
     * @param filename The base filename for loading state files.
     * @return True if successful, false otherwise.
     */
    bool loadState(const std::string& filename);

    const std::vector<DecisionRecord>& getDecisionHistory() const { return decision_history_; }
    Portfolio& getPortfolio() { return portfolio_ref_; } // Return non-const reference
    const Portfolio& getPortfolio() const { return portfolio_ref_; } // Return const reference
    double getCumulativeReward() const { return cumulative_reward_; }
    double getEpsilon() const { return epsilon_; }
    bool isInitialized() const { return is_initialized_; } 

private:
    // Helper methods for decision making and state management
    TradingDecision _determineActionFromSignal(double prediction_signal, double& confidence);
    
    /**
     * @brief Extracts market features from the given PriceTick.
     * @param price_tick The current PriceTick data.
     * @return A map of feature names to their values.
     */
    std::map<std::string, double> _getCurrentMarketFeatures(const PriceTick& price_tick);
    
    double _determineQuantity(double current_price, TradingDecision decision, double confidence);
    void _logDecision(const DecisionRecord& record); 

    std::string symbol_;
    CoinbaseAdvancedTradeApi* coinbase_api_ptr_; 
    Portfolio& portfolio_ref_; // Changed from owned Portfolio to reference
    NeuralNetworkInterface nn_interface_;
    TechnicalAnalysis tech_analyzer_; 

    std::vector<PriceTick> price_series_;
    size_t max_price_series_size_; 

    std::vector<DecisionRecord> decision_history_;
    std::vector<double> reward_history_;
    double cumulative_reward_;
    double epsilon_; 
    double last_decision_price_; 
    static constexpr double EPSILON_DECAY = 0.995;

    bool is_initialized_; 
    std::ofstream log_file_;
    std::stringstream decision_log_buffer_; 

    // These seem like older/alternative helper methods, review if still needed or should be removed/merged.
    // TradingDecision _determineAction(double predicted_price, double current_price, double& confidence);
    // double _calculateConfidence(double predicted_price, double current_price);
    // double _determineQuantity(TradingDecision decision, double confidence, double current_price); // Duplicate signature, one above is (double, TD, double)
};

} // namespace NeuroGen
#endif // NEUROGEN_AUTONOMOUSTRADINGAGENT_H