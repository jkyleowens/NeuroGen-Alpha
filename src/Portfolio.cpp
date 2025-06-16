#include "Portfolio.h" // Correct include path
#include <iostream>
#include <iomanip> 
#include <stdexcept> 
#include <fstream> // Required for file operations
#include <nlohmann/json.hpp> // For JSON serialization

// Private helper method to update the initial value of the portfolio
void Portfolio::_updateInitialValue(double initial_cash_for_value, double initial_coins_for_value, double price_for_initial_coins) {
    if (initial_coins_for_value > 0 && price_for_initial_coins > 0) {
        initial_value_ = initial_cash_for_value + (initial_coins_for_value * price_for_initial_coins);
    } else {
        // If no initial coins or no price for them, initial value is just the cash
        initial_value_ = initial_cash_for_value;
    }
    // std::cout << "[Portfolio] Initial portfolio value set to: $" << std::fixed << std::setprecision(2) << initial_value_ << std::endl;
}

Portfolio::Portfolio(double initial_cash, double initial_coins)
    : cash_balance_(initial_cash), coin_balance_(initial_coins) {
    if (initial_cash < 0) {
        // std::cerr << "[Portfolio] Warning: Initial cash is negative. Setting to 0." << std::endl;
        cash_balance_ = 0.0;
    }
    if (initial_coins < 0) {
        // std::cerr << "[Portfolio] Warning: Initial coins is negative. Setting to 0." << std::endl;
        coin_balance_ = 0.0;
    }
    // To accurately set initial_value_ if initial_coins > 0, we'd need an initial price.
    // For now, if initial_coins are passed, their value isn't part of initial_value_ unless a mechanism to pass their price is added.
    // The _updateInitialValue method expects a price. Let's assume 0 if not provided, meaning initial_value_ will be initial_cash.
    _updateInitialValue(cash_balance_, coin_balance_, 0.0); 
    std::cout << "[Portfolio] Initialized. Cash: $" << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::fixed << std::setprecision(8) << coin_balance_ 
              << ". Initial Portfolio Value: $" << std::fixed << std::setprecision(2) << initial_value_ << std::endl;
}

bool Portfolio::executeBuy(double quantity, double price) {
    if (quantity <= 0.0) {
        std::cerr << "[Portfolio] Buy Error: Quantity must be positive. Attempted: " << quantity << std::endl;
        return false;
    }
    if (price <= 0.0) {
        std::cerr << "[Portfolio] Buy Error: Price must be positive. Attempted: " << price << std::endl;
        return false;
    }

    double cost = quantity * price;

    if (cost > cash_balance_) {
        std::cout << "[Portfolio] Insufficient funds for buy. Cost: $" << std::fixed << std::setprecision(2) << cost
                  << ", Available Cash: $" << std::fixed << std::setprecision(2) << cash_balance_ << ". Order rejected." << std::endl;
        // Optional: Implement partial buy if desired, for now, reject.
        // If partial buy:
        // quantity = cash_balance_ / price; // buy max possible
        // cost = quantity * price;
        // if (quantity <= 1e-8) { // Check for very small or zero quantity
        //     std::cout << "[Portfolio] Cannot buy any coins with available cash." << std::endl;
        //     return false;
        // }
        // std::cout << "[Portfolio] Adjusting buy quantity to " << std::fixed << std::setprecision(8) << quantity << " due to insufficient funds." << std::endl;
        return false; // Reject if cannot afford the full requested quantity
    }

    cash_balance_ -= cost;
    coin_balance_ += quantity;

    std::cout << "[Portfolio] BUY: " << std::fixed << std::setprecision(8) << quantity
              << " coins @ $" << std::fixed << std::setprecision(2) << price
              << " (Cost: $" << std::fixed << std::setprecision(2) << cost << ")" << std::endl;
    std::cout << "[Portfolio] New Balance -> Cash: $" << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::fixed << std::setprecision(8) << coin_balance_ << std::endl;
    return true;
}

bool Portfolio::executeSell(double quantity, double price) {
    if (quantity <= 0.0) {
        std::cerr << "[Portfolio] Sell Error: Quantity must be positive. Attempted: " << quantity << std::endl;
        return false;
    }
    if (price <= 0.0) {
        std::cerr << "[Portfolio] Sell Error: Price must be positive. Attempted: " << price << std::endl;
        return false;
    }

    if (quantity > coin_balance_) {
        std::cout << "[Portfolio] Insufficient coins for sell. Requested: " << std::fixed << std::setprecision(8) << quantity
                  << ", Available Coins: " << std::fixed << std::setprecision(8) << coin_balance_ << ". Order rejected." << std::endl;
        // Optional: Implement partial sell if desired, for now, reject.
        // If partial sell:
        // quantity = coin_balance_; // sell all available
        // if (quantity <= 1e-8) { // Check for very small or zero quantity
        //     std::cout << "[Portfolio] No coins to sell." << std::endl;
        //     return false;
        // }
        // std::cout << "[Portfolio] Adjusting sell quantity to " << std::fixed << std::setprecision(8) << quantity << " due to insufficient coins." << std::endl;
        return false; // Reject if cannot sell the full requested quantity
    }

    double revenue = quantity * price;
    cash_balance_ += revenue;
    coin_balance_ -= quantity;

    std::cout << "[Portfolio] SELL: " << std::fixed << std::setprecision(8) << quantity
              << " coins @ $" << std::fixed << std::setprecision(2) << price
              << " (Revenue: $" << std::fixed << std::setprecision(2) << revenue << ")" << std::endl;
    std::cout << "[Portfolio] New Balance -> Cash: $" << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::fixed << std::setprecision(8) << coin_balance_ << std::endl;
    return true;
}

double Portfolio::getCashBalance() const {
    return cash_balance_;
}

double Portfolio::getCoinBalance() const {
    return coin_balance_;
}

double Portfolio::getCurrentValue(double current_price) const {
    if (current_price < 0.0) {
        std::cerr << "[Portfolio] Warning: Current price is negative (" << current_price << ") for calculating current value. Using 0.0." << std::endl;
        current_price = 0.0;
    }
    return cash_balance_ + (coin_balance_ * current_price);
}

double Portfolio::getProfitAndLoss(double current_price) const {
    if (current_price < 0.0) {
        // Match behavior of getCurrentValue for negative price
        current_price = 0.0; 
    }
    double current_total_value = getCurrentValue(current_price);
    return current_total_value - initial_value_;
}

void Portfolio::reset(double new_initial_cash, double new_initial_coins) {
    if (new_initial_cash < 0) {
        // std::cerr << "[Portfolio] Warning: Reset initial cash is negative. Setting to 0." << std::endl;
        new_initial_cash = 0.0;
    }
    if (new_initial_coins < 0) {
        // std::cerr << "[Portfolio] Warning: Reset initial coins is negative. Setting to 0." << std::endl;
        new_initial_coins = 0.0;
    }
    cash_balance_ = new_initial_cash;
    coin_balance_ = new_initial_coins;
    _updateInitialValue(cash_balance_, coin_balance_, 0.0); 
    std::cout << "[Portfolio] Reset. Cash: $" << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::fixed << std::setprecision(8) << coin_balance_
              << ". New Initial Portfolio Value: $" << std::fixed << std::setprecision(2) << initial_value_ << std::endl;
}

bool Portfolio::saveState(const std::string& filename) const {
    std::ofstream state_file(filename);
    if (!state_file.is_open()) {
        std::cerr << "[Portfolio] Error: Could not open state file for saving: " << filename << std::endl;
        return false;
    }

    nlohmann::json j;
    j["cash_balance"] = cash_balance_;
    j["coin_balance"] = coin_balance_;
    j["initial_value"] = initial_value_;

    state_file << j.dump(4); // Dump with an indent of 4 for readability
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
        cash_balance_ = j.at("cash_balance").get<double>();
        coin_balance_ = j.at("coin_balance").get<double>();
        initial_value_ = j.at("initial_value").get<double>();
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[Portfolio] Error: Failed to parse portfolio state file " << filename << ". Error: " << e.what() << std::endl;
        state_file.close();
        return false;
    }

    state_file.close();
    std::cout << "[Portfolio] State loaded from " << filename << std::endl;
    std::cout << "[Portfolio] Loaded Balance -> Cash: $" << std::fixed << std::setprecision(2) << cash_balance_
              << ", Coins: " << std::fixed << std::setprecision(8) << coin_balance_ 
              << ". Initial Portfolio Value: $" << std::fixed << std::setprecision(2) << initial_value_ << std::endl;
    return true;
}
