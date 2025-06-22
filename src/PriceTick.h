#ifndef NEUROGEN_PRICETICK_H
#define NEUROGEN_PRICETICK_H

#include <map>
#include <string>

// Represents a single point in a time series of market data.
struct PriceTick {
    long timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    // Added indicators map to store flexible additional data like TA values.
    std::map<std::string, double> indicators;

    // Default constructor
    PriceTick() : timestamp(0), open(0.0), high(0.0), low(0.0), close(0.0), volume(0.0) {}

    // Parameterized constructor for core OHLCV data
    PriceTick(long ts, double o, double h, double l, double c, double v)
        : timestamp(ts), open(o), high(h), low(l), close(c), volume(v) {}

    // Basic validation to ensure data integrity
    bool validate() const {
        return timestamp > 0 && open > 0 && high > 0 && low > 0 && close > 0 && volume >= 0;
    }
};

#endif // NEUROGEN_PRICETICK_H
