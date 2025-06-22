#ifndef NEUROGEN_SIMULATION_H
#define NEUROGEN_SIMULATION_H

#include <string>
#include <vector>
#include <fstream>
#include "NeuroGen/PriceTick.h"
#include "NeuroGen/AutonomousTradingAgent.h"

// Forward declarations for classes used in the header
namespace NeuroGen {
    class Portfolio;
    class CoinbaseAdvancedTradeApi;
}

namespace NeuroGen {

class Simulation {
public:
    Simulation(
        AutonomousTradingAgent& agent,
        Portfolio& portfolio,
        const std::vector<PriceTick>& initial_price_data,
        CoinbaseAdvancedTradeApi* api_client = nullptr);
    ~Simulation();

    void run(int max_ticks = 0);
    bool advanceTick();
    bool isFinished() const;

    // Getter methods
    int getCurrentTickIndex() const;
    const std::vector<PriceTick>& getTimeSeriesData() const;
    const AutonomousTradingAgent& getAgent() const;
    const Portfolio& getPortfolio() const;

    bool saveState(const std::string& filename_base);
    bool loadState(const std::string& filename_base);

private:
    AutonomousTradingAgent& agent_;
    Portfolio& portfolio_;
    std::vector<PriceTick> time_series_data_;
    int current_tick_index_;
    CoinbaseAdvancedTradeApi* coinbase_api_ptr_;
    bool is_running_;
    std::ofstream simulation_log_;

    double _calculateReward(
        const AutonomousTradingAgent::DecisionRecord& decision_details,
        const PriceTick& tick_at_decision,
        const PriceTick* tick_after_decision,
        double portfolio_value_before_trade,
        double portfolio_value_after_trade
    );
    
    void _logSimulationStep(
        int tick_index,
        const PriceTick& current_tick_data,
        double portfolio_value,
        double reward,
        const AutonomousTradingAgent::DecisionRecord& decision_details
    );
};

} // namespace NeuroGen

#endif // NEUROGEN_SIMULATION_H