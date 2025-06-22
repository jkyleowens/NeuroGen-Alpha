#include <NeuroGen/Portfolio.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>

namespace NeuroGen {

// Corrected constructor to match header declaration
Portfolio::Portfolio(const std::string& asset_symbol, double initial_cash)
    : asset_symbol_(asset_symbol),
      cash_balance_(initial_cash),
      coin_balance_(0.0),
      initial_cash_(initial_cash),
      initial_portfolio_value_(initial_cash),
      trade_count_(0) {
    if (initial_cash < 0) {
        std::cerr << "[Portfolio] Warning: Initial cash is negative. Setting to 0." << std::endl;
        initial_cash_ = 0.0;
        cash_balance_ = 0.0;
        initial_portfolio_value_ = 0.0;
    }

    // Open log file
    std::string log_filename = "portfolio_" + asset_symbol_ + "_tradelog.csv";
    trade_log_file_.open(log_filename, std::ios::out | std::ios::app);
    if (!trade_log_file_.is_open()) {
        std::cerr << "[Portfolio] Warning: Could not open trade log file: " << log_filename << std::endl;
    } else {
        // Write header if the file is new/empty
        trade_log_file_.seekp(0, std::ios::end);
        if (trade_log_file_.tellp() == 0) {
            trade_log_file_ << "Timestamp,Type,Quantity,Price,Fee,TotalCost,CashBalance,CoinBalance" << std::endl;
        }
    }

    std::cout << "[Portfolio] Initialized for asset: " << asset_symbol_ 
              << " with initial cash: " << std::fixed << std::setprecision(2) << initial_cash_
              << ". Initial Portfolio Value: " << initial_portfolio_value_ << std::endl;
}

Portfolio::~Portfolio() {
    if (trade_log_file_.is_open()) {
        trade_log_file_.close();
    }
    std::cout << "[Portfolio] Destroyed for asset: " << asset_symbol_ << ". Final cash: " 
              << std::fixed << std::setprecision(2) << cash_balance_ 
              << ", Final assets: " << coin_balance_ << std::endl;
}

bool Portfolio::executeBuy(double quantity, double price) {
    if (quantity <= 0 || price <= 0) {
        std::cerr << "[Portfolio] Buy Error: Quantity and price must be positive." << std::endl;
        return false;
    }

    double total_cost = quantity * price;
    // Assuming a simple fee structure for now
    double fee = 0.0; // No fee for this example, can be changed

    if (cash_balance_ < total_cost + fee) {
        std::cerr << "[Portfolio] Insufficient cash to BUY " << quantity << " " << asset_symbol_
                  << " (Required: " << std::fixed << std::setprecision(2) << (total_cost + fee)
                  << ", Available: " << cash_balance_ << "). Trade rejected." << std::endl;
        return false;
    }

    cash_balance_ -= (total_cost + fee);
    coin_balance_ += quantity;
    trade_count_++;

    TradeRecord record = {
        std::chrono::system_clock::now(),
        TradeType::BUY,
        quantity,
        price,
        fee,
        total_cost
    };
    _logTrade(record);
    return true;
}

bool Portfolio::executeSell(double quantity, double price) {
    if (quantity <= 0 || price <= 0) {
        std::cerr << "[Portfolio] Sell Error: Quantity and price must be positive." << std::endl;
        return false;
    }
    
    if (coin_balance_ < quantity) {
        std::cerr << "[Portfolio] Insufficient assets to SELL " << quantity << " " << asset_symbol_
                  << " (Required: " << quantity 
                  << ", Available: " << coin_balance_ << "). Trade rejected." << std::endl;
        return false;
    }

    double total_value = quantity * price;
    double fee = 0.0; // No fee for this example

    cash_balance_ += (total_value - fee);
    coin_balance_ -= quantity;
    trade_count_++;

    TradeRecord record = {
        std::chrono::system_clock::now(),
        TradeType::SELL,
        quantity,
        price,
        fee,
        total_value
    };
    _logTrade(record);
    return true;
}

void Portfolio::_logTrade(const TradeRecord& record) {
    trade_history_.push_back(record);
    
    if (trade_log_file_.is_open()) {
        std::time_t time_now = std::chrono::system_clock::to_time_t(record.timestamp);
        std::tm tm_now = *std::gmtime(&time_now);
        trade_log_file_ << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ") << ","
                        << (record.type == TradeType::BUY ? "BUY" : "SELL") << ","
                        << std::fixed << std::setprecision(8) << record.quantity << ","
                        << std::setprecision(2) << record.price << ","
                        << std::setprecision(4) << record.fee << ","
                        << std::setprecision(2) << record.total_cost << ","
                        << cash_balance_ << ","
                        << std::setprecision(8) << coin_balance_ << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2)
              << "[Portfolio] Logged Trade: " << (record.type == TradeType::BUY ? "BUY" : "SELL")
              << " " << std::setprecision(8) << record.quantity << " " << asset_symbol_ 
              << " @ " << std::setprecision(2) << record.price
              << ". New Cash: " << cash_balance_ << ", New Coins: " << std::setprecision(8) << coin_balance_ << std::endl;
}

double Portfolio::getCurrentValue(double current_asset_price) const {
    if (current_asset_price < 0) {
        std::cerr << "[Portfolio] Warning: current_asset_price is negative. Using 0 for asset value calculation." << std::endl;
        return cash_balance_;
    }
    return cash_balance_ + (coin_balance_ * current_asset_price);
}

double Portfolio::getCashBalance() const {
    return cash_balance_;
}

double Portfolio::getCoinBalance() const {
    return coin_balance_;
}

const std::string& Portfolio::getAssetSymbol() const {
    return asset_symbol_;
}

int Portfolio::getTradeCount() const {
    return trade_count_;
}

double Portfolio::getInitialCash() const {
    return initial_cash_;
}

double Portfolio::getInitialPortfolioValue() const {
    return initial_portfolio_value_;
}

const std::vector<Portfolio::TradeRecord>& Portfolio::getTradeHistory() const {
    return trade_history_;
}

void Portfolio::reset(double new_initial_cash) {
    if (new_initial_cash < 0) {
        std::cerr << "[Portfolio] Warning: Reset initial cash is negative. Setting to 0." << std::endl;
        new_initial_cash = 0.0;
    }
    initial_cash_ = new_initial_cash;
    cash_balance_ = initial_cash_;
    coin_balance_ = 0.0;
    trade_count_ = 0;
    trade_history_.clear();
    initial_portfolio_value_ = initial_cash_;
    
    std::cout << "[Portfolio] Reset for asset: " << asset_symbol_ 
              << " with initial cash: " << std::fixed << std::setprecision(2) << initial_cash_ 
              << ". New Initial Portfolio Value: " << initial_portfolio_value_ << std::endl;
}

bool Portfolio::saveState(const std::string& filename) const {
    std::ofstream state_file(filename);
    if (!state_file.is_open()) {
        std::cerr << "[Portfolio] Error: Could not open state file for saving: " << filename << std::endl;
        return false;
    }

    nlohmann::json j;
    j["asset_symbol"] = asset_symbol_;
    j["initial_cash"] = initial_cash_;
    j["cash_balance"] = cash_balance_;
    j["coin_balance"] = coin_balance_;
    j["initial_portfolio_value"] = initial_portfolio_value_;
    j["trade_count"] = trade_count_;
    
    state_file << j.dump(4); 
    state_file.close();

    std::cout << "[Portfolio] State saved to " << filename << std::endl;
    return state_file.good();
}

bool Portfolio::loadState(const std::string& filename) {
    std::ifstream state_file(filename);
    if (!state_file.is_open()) {
        std::cerr << "[Portfolio] Error: Could not open state file for loading: " << filename << std::endl;
        return false;
    }

    nlohmann::json j;
    try {
        state_file >> j;
        asset_symbol_ = j.at("asset_symbol").get<std::string>();
        initial_cash_ = j.at("initial_cash").get<double>();
        cash_balance_ = j.at("cash_balance").get<double>();
        coin_balance_ = j.at("coin_balance").get<double>();
        initial_portfolio_value_ = j.at("initial_portfolio_value").get<double>();
        trade_count_ = j.at("trade_count").get<int>();
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[Portfolio] Error: Failed to parse portfolio state file " << filename << ". Error: " << e.what() << std::endl;
        state_file.close();
        return false;
    }

    state_file.close();
    std::cout << "[Portfolio] State loaded from " << filename << std::endl;
    std::cout << "[Portfolio] Loaded State -> Asset: " << asset_symbol_
              << ", Cash: " << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::setprecision(8) << coin_balance_ 
              << ". Initial Portfolio Value: " << initial_portfolio_value_ << std::endl;
    return true;
}

} // namespace NeuroGen
