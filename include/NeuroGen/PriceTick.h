#ifndef NEUROGEN_PRICETICK_H
#define NEUROGEN_PRICETICK_H

#include <string> // For potential future use, e.g. symbol
#include <limits> // For std::numeric_limits

/**
 * @brief Represents a single price tick, typically an OHLCV (Open, High, Low, Close, Volume) data point.
 */
struct PriceTick {
    long timestamp = 0;    // Unix timestamp (seconds since epoch)
    double open = 0.0;     // Opening price
    double high = 0.0;     // Highest price
    double low = 0.0;      // Lowest price
    double close = 0.0;    // Closing price
    double volume = 0.0;   // Trading volume

    /**
     * @brief Default constructor.
     */
    PriceTick() = default;

    /**
     * @brief Parameterized constructor.
     * @param ts Timestamp.
     * @param o Open price.
     * @param h High price.
     * @param l Low price.
     * @param c Close price.
     * @param v Volume.
     */
    PriceTick(long ts, double o, double h, double l, double c, double v)
        : timestamp(ts), open(o), high(h), low(l), close(c), volume(v) {}

    /**
     * @brief Validates the PriceTick data.
     * Checks for non-negative prices and volume, and ensures high >= low,
     * high >= open, high >= close, low <= open, low <= close.
     * @return True if the data is valid, false otherwise.
     */
    bool validate() const {
        if (timestamp < 0 || open < 0.0 || high < 0.0 || low < 0.0 || close < 0.0 || volume < 0.0) {
            return false;
        }
        if (low > high) {
            return false;
        }
        // Allow some tolerance for floating point comparisons if necessary,
        // but for basic validation, direct comparison is often sufficient.
        if (open > high || open < low || close > high || close < low) {
            // This check ensures open and close are within the high-low range.
            // Depending on the data source, open/close might exactly equal high/low.
            // A very small epsilon could be used if strict equality is too rigid.
            // For now, assuming open/close must be strictly within or equal to H/L.
            constexpr double epsilon = 1e-9; // A small tolerance
            if (open > high + epsilon || open < low - epsilon ||
                close > high + epsilon || close < low - epsilon) {
                return false;
            }
        }
        return true;
    }
};

#endif // NEUROGEN_PRICETICK_H
