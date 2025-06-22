#ifndef NEUROGEN_PORTFOLIO_H
#define NEUROGEN_PORTFOLIO_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace NeuroGen {

class Portfolio {
public:
    // Enum to define trade types clearly
    enum class TradeType {
        BUY,
        SELL
    };

    // Struct to record each trade's details
    struct TradeRecord {
        std::chrono::system_clock::time_point timestamp;
        TradeType type;
        double quantity;
        double price;
        double fee;
        double total_cost;
    };

    // **FIXED CONSTRUCTOR**: Now takes the asset symbol.
    Portfolio(const std::string& asset_symbol, double initial_cash);
    ~Portfolio();

    // Public methods for trade execution
    bool executeBuy(double quantity, double price);
    bool executeSell(double quantity, double price);
    
    // **FIXED DECLARATIONS**: Added all necessary 'getter' methods.
    double getCurrentValue(double current_asset_price) const;
    double getCashBalance() const;
    double getCoinBalance() const; // Renamed from getAssetQuantity for clarity
    const std::string& getAssetSymbol() const;
    int getTradeCount() const;
    double getInitialCash() const;
    double getInitialPortfolioValue() const;
    const std::vector<TradeRecord>& getTradeHistory() const;

    // Public methods for state management
    void reset(double new_initial_cash);
    bool saveState(const std::string& filename) const;
    bool loadState(const std::string& filename);

private:
    // **FIXED DECLARATIONS**: Added all necessary member variables.
    std::string asset_symbol_;
    double cash_balance_;
    double coin_balance_; // Renamed from asset_quantity_

    double initial_cash_;
    double initial_portfolio_value_;
    int trade_count_;

    std::vector<TradeRecord> trade_history_;
    std::ofstream trade_log_file_;

    void _logTrade(const TradeRecord& record);
};

} // namespace NeuroGen
#endif // NEUROGEN_PORTFOLIO_H
