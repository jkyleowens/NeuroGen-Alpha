#ifndef NEUROGEN_PORTFOLIO_H
#define NEUROGEN_PORTFOLIO_H

#include <string> // For potential future use, e.g. currency symbols
#include <fstream> // For saving/loading state

class Portfolio {
public:
    /**
     * @brief Constructor.
     * @param initial_cash The starting cash balance.
     * @param initial_coins The starting coin balance (optional, defaults to 0).
     */
    Portfolio(double initial_cash, double initial_coins = 0.0);

    /**
     * @brief Executes a buy order.
     * Decreases cash balance and increases coin balance.
     * @param quantity The amount of coin to buy.
     * @param price The price per coin.
     * @return True if the buy order was successful (e.g., sufficient cash), false otherwise.
     */
    bool executeBuy(double quantity, double price);

    /**
     * @brief Executes a sell order.
     * Increases cash balance and decreases coin balance.
     * @param quantity The amount of coin to sell.
     * @param price The price per coin.
     * @return True if the sell order was successful (e.g., sufficient coins), false otherwise.
     */
    bool executeSell(double quantity, double price);

    /**
     * @brief Gets the current cash balance.
     * @return The current cash balance.
     */
    double getCashBalance() const;

    /**
     * @brief Gets the current coin balance.
     * @return The current coin balance.
     */
    double getCoinBalance() const;

    /**
     * @brief Calculates the current total value of the portfolio.
     * @param current_price The current market price of the coin.
     * @return The total value (cash_balance + coin_balance * current_price).
     */
    double getCurrentValue(double current_price) const;

    /**
     * @brief Calculates the profit and loss (P&L) relative to the initial portfolio value.
     * @param current_price The current market price of the coin.
     * @return The profit or loss.
     */
    double getProfitAndLoss(double current_price) const;

    /**
     * @brief Resets the portfolio to an initial state.
     * @param initial_cash The new starting cash balance.
     * @param initial_coins The new starting coin balance (optional, defaults to 0).
     */
    void reset(double initial_cash, double initial_coins = 0.0);

    /**
     * @brief Saves the portfolio's state to a file.
     * @param filename The filename to save the state to.
     * @return True if successful, false otherwise.
     */
    bool saveState(const std::string& filename) const;

    /**
     * @brief Loads the portfolio's state from a file.
     * @param filename The filename to load the state from.
     * @return True if successful, false otherwise.
     */
    bool loadState(const std::string& filename);

private:
    double cash_balance_;
    double coin_balance_;
    double initial_value_;

    /**
     * @brief Updates the initial value of the portfolio. Called upon initialization or reset.
     *        For simplicity, this might be based on initial cash only if starting with no coins,
     *        or initial_cash + initial_coins * some_initial_price if starting with coins.
     *        The design doc implies initialValue is set once. If starting with coins,
     *        an initial price for those coins would be needed to set initial_value_ accurately.
     *        For now, let's assume initial_value_ is primarily based on initial_cash_ if no initial_coins_price is given.
     */
    void _updateInitialValue(double initial_cash_for_value, double initial_coins_for_value, double price_for_initial_coins = 0.0);
};

#endif // NEUROGEN_PORTFOLIO_H
