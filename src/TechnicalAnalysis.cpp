#include "NeuroGen/TechnicalAnalysis.h" // Corrected include
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <tuple> // Added for std::tuple, std::make_tuple, std::get

TechnicalAnalysis::TechnicalAnalysis(const std::vector<PriceTick>& price_series)
    : price_series_ptr_(&price_series) { // Corrected to initialize price_series_ptr_
}

std::map<std::string, double> TechnicalAnalysis::getFeaturesForTick(int index) {
    std::map<std::string, double> features;
    
    // Validate index
    if (index < 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_
        std::cerr << "Error: Index out of bounds in TechnicalAnalysis::getFeaturesForTick" << std::endl;
        return features;
    }
    
    // Basic price features
    features["price"] = (*price_series_ptr_)[index].close; // Used price_series_ptr_
    features["volume"] = (*price_series_ptr_)[index].volume; // Used price_series_ptr_
    features["price_change"] = (*price_series_ptr_)[index].close - (*price_series_ptr_)[index].open; // Used price_series_ptr_
    features["price_change_pct"] = ((*price_series_ptr_)[index].close / (*price_series_ptr_)[index].open) - 1.0; // Used price_series_ptr_
    
    // Calculate technical indicators if we have enough history
    if (index >= 14) { // Assuming 14 is a reasonable minimum, specific indicators might need more
        // Moving Averages
        features["sma_5"] = _calculateSMA(5, index);
        features["sma_10"] = _calculateSMA(10, index);
        features["sma_20"] = _calculateSMA(20, index);
        // Check for sufficient data for longer SMAs
        if (index >= 50) features["sma_50"] = _calculateSMA(50, index);
        else features["sma_50"] = (*price_series_ptr_)[index].close; // Default or NaN could be better
        if (index >= 200) features["sma_200"] = _calculateSMA(200, index);
        else features["sma_200"] = (*price_series_ptr_)[index].close; // Default or NaN
        
        // EMA
        features["ema_12"] = _calculateEMA(12, index);
        features["ema_26"] = _calculateEMA(26, index);
        
        // MACD
        auto macd = _calculateMACD(index);
        features["macd"] = macd.first;
        features["macd_signal"] = macd.second;
        features["macd_histogram"] = macd.first - macd.second;
        
        // RSI
        features["rsi_14"] = _calculateRSI(14, index);
        
        // Bollinger Bands
        if (index >= 20) { // Bollinger Bands typically use a 20-period SMA
            auto bb = _calculateBollingerBands(20, index, 2.0);
            features["bb_upper"] = std::get<0>(bb); // Corrected access for std::tuple
            features["bb_middle"] = std::get<1>(bb); // Corrected access for std::tuple
            features["bb_lower"] = std::get<2>(bb); // Corrected access for std::tuple
            if (std::get<1>(bb) != 0) { // Avoid division by zero for bb_width
                 features["bb_width"] = (std::get<0>(bb) - std::get<2>(bb)) / std::get<1>(bb);
            } else {
                 features["bb_width"] = 0;
            }
            if ((std::get<0>(bb) - std::get<2>(bb)) != 0) { // Avoid division by zero for bb_position
                features["bb_position"] = ((*price_series_ptr_)[index].close - std::get<2>(bb)) / (std::get<0>(bb) - std::get<2>(bb)); // Used price_series_ptr_
            } else {
                features["bb_position"] = 0.5; // Or some other sensible default
            }
        } else {
            // Default values if not enough data for Bollinger Bands
            features["bb_upper"] = (*price_series_ptr_)[index].close;
            features["bb_middle"] = (*price_series_ptr_)[index].close;
            features["bb_lower"] = (*price_series_ptr_)[index].close;
            features["bb_width"] = 0;
            features["bb_position"] = 0.5;
        }
        
        // ATR - Average True Range (volatility)
        features["atr_14"] = _calculateATR(14, index);
        
        // OBV - On-Balance Volume
        features["obv"] = _calculateOBV(index);
        
        // Stochastic Oscillator
        features["stoch_14"] = _calculateStochastic(14, index);
        
        // Price momentum
        if (index >= 5) features["momentum_5"] = (*price_series_ptr_)[index].close - (*price_series_ptr_)[index-5].close; // Used price_series_ptr_
        else features["momentum_5"] = 0;
        if (index >= 10) features["momentum_10"] = (*price_series_ptr_)[index].close - (*price_series_ptr_)[index-10].close; // Used price_series_ptr_
        else features["momentum_10"] = 0;
        if (index >= 20) {
            features["momentum_20"] = (*price_series_ptr_)[index].close - (*price_series_ptr_)[index-20].close; // Used price_series_ptr_
        } else {
            features["momentum_20"] = 0;
        }
        
        // Moving average crossovers
        // Ensure components are calculated before trying to use them
        if (features.count("sma_5") && features.count("sma_10"))
            features["sma_5_10_cross"] = features["sma_5"] - features["sma_10"];
        else features["sma_5_10_cross"] = 0;

        if (features.count("sma_10") && features.count("sma_20"))
            features["sma_10_20_cross"] = features["sma_10"] - features["sma_20"];
        else features["sma_10_20_cross"] = 0;

        if (features.count("sma_50") && features.count("sma_200"))
            features["sma_50_200_cross"] = features["sma_50"] - features["sma_200"];
        else features["sma_50_200_cross"] = 0;
        
        // Volatility
        if (index >= 20) {
            auto prices = _getPrices(index - 19, index); // Corrected range for 20 periods (index-19 to index inclusive)
            if (prices.size() == 20) { // Ensure we have 20 prices
                double mean = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
                if (mean != 0) { // Avoid division by zero for volatility_20
                    double sq_sum = std::inner_product(prices.begin(), prices.end(), prices.begin(), 0.0);
                    double std_dev = std::sqrt(sq_sum / prices.size() - mean * mean);
                    features["volatility_20"] = std_dev / mean;
                } else {
                    features["volatility_20"] = 0;
                }
            } else {
                 features["volatility_20"] = 0; // Not enough data
            }
        } else {
            features["volatility_20"] = 0;
        }
    } else {
        // Default values if not enough history for most indicators
        features["sma_5"] = (*price_series_ptr_)[index].close;
        features["sma_10"] = (*price_series_ptr_)[index].close;
        features["sma_20"] = (*price_series_ptr_)[index].close;
        features["sma_50"] = (*price_series_ptr_)[index].close;
        features["sma_200"] = (*price_series_ptr_)[index].close;
        features["ema_12"] = (*price_series_ptr_)[index].close;
        features["ema_26"] = (*price_series_ptr_)[index].close;
        features["macd"] = 0;
        features["macd_signal"] = 0;
        features["macd_histogram"] = 0;
        features["rsi_14"] = 50.0; // Neutral RSI
        features["bb_upper"] = (*price_series_ptr_)[index].close;
        features["bb_middle"] = (*price_series_ptr_)[index].close;
        features["bb_lower"] = (*price_series_ptr_)[index].close;
        features["bb_width"] = 0;
        features["bb_position"] = 0.5;
        features["atr_14"] = 0;
        if (!(*price_series_ptr_).empty()) features["obv"] = (*price_series_ptr_)[index].volume; else features["obv"] = 0;
        features["stoch_14"] = 50.0; // Neutral Stochastic
        features["momentum_5"] = 0;
        features["momentum_10"] = 0;
        features["momentum_20"] = 0;
        features["sma_5_10_cross"] = 0;
        features["sma_10_20_cross"] = 0;
        features["sma_50_200_cross"] = 0;
        features["volatility_20"] = 0;
    }
    
    return features;
}

void TechnicalAnalysis::updatePriceSeries(const std::vector<PriceTick>& price_series) {
    price_series_ptr_ = &price_series; // Corrected to assign to price_series_ptr_
}

// Public wrapper methods for calculating indicators
void TechnicalAnalysis::calculateSMA(const std::vector<PriceTick>& price_series, [[maybe_unused]] int period) { // Added [[maybe_unused]]
    price_series_ptr_ = &price_series;
}

void TechnicalAnalysis::calculateEMA(const std::vector<PriceTick>& price_series, [[maybe_unused]] int period) { // Added [[maybe_unused]]
    price_series_ptr_ = &price_series;
}

void TechnicalAnalysis::calculateRSI(const std::vector<PriceTick>& price_series, [[maybe_unused]] int period) { // Added [[maybe_unused]]
    price_series_ptr_ = &price_series;
}

double TechnicalAnalysis::_calculateSMA(int period, int index) const { // Added const
    if (index < period - 1 || period <= 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_ and added boundary check
        if (!(*price_series_ptr_).empty() && index < static_cast<int>((*price_series_ptr_).size()) && index >=0) return (*price_series_ptr_)[index].close;
        return 0.0; // Or handle error appropriately
    }
    
    double sum = 0.0;
    for (int i = index - period + 1; i <= index; i++) {
        sum += (*price_series_ptr_)[i].close; // Used price_series_ptr_
    }
    
    return sum / period;
}

double TechnicalAnalysis::_calculateEMA(int period, int index) const { // Added const
    if (index < 0 || index >= static_cast<int>((*price_series_ptr_).size())) return 0.0; // Boundary check

    // EMA requires at least 'period' data points for a meaningful start with SMA.
    // If index < period -1, SMA cannot be calculated for index - period.
    // A common approach is to use SMA of the first 'period' elements for the EMA at index 'period-1'.
    // Then iteratively calculate EMA.
    // The current _calculateSMA(period, index - period) might be problematic if index - period is too small.
    // For simplicity, if not enough data for the initial SMA for EMA, return current price.
    // A more robust EMA calculation might be needed for production.
    if (index < period -1 || period <=0 ) { // if index < period, we can't get SMA for index-period.
         if (!(*price_series_ptr_).empty()) return (*price_series_ptr_)[index].close;
         return 0.0;
    }
    if (index < period) { // Not enough data to calculate SMA for index-period. Let's say EMA starts at index = period-1 with SMA(period, period-1)
        // This part needs careful thought for correct EMA initialization.
        // For now, let's assume _calculateSMA handles small indices by returning current price or similar.
        // The original code had: if (index < period - 1 || period <= 0) return price_series_[index].close;
        // And then: double ema = _calculateSMA(period, index - period);
        // If index = period-1, then index-period = -1. _calculateSMA needs to handle this.
        // Let's assume _calculateSMA returns a sensible value or current price for out-of-range/insufficient data.
    }


    // Start with SMA for the initial EMA value.
    // The period for this initial SMA should be 'period', and it should be calculated at an earlier point.
    // For EMA at 'index', the previous EMA is at 'index-1'.
    // Let's adjust the logic slightly for clarity, though the original recursive-like structure is common.
    // A common way: EMA_today = Price_today * multiplier + EMA_yesterday * (1-multiplier)
    // Initial EMA can be a simple moving average.

    std::vector<double> prices;
    for(int i=0; i<=index; ++i) prices.push_back((*price_series_ptr_)[i].close);

    if (prices.size() < static_cast<size_t>(period)) { // Not enough data
        if (!prices.empty()) return prices.back();
        return 0.0;
    }

    double ema = 0.0;
    // Calculate initial SMA for the first 'period' prices
    for(int i=0; i<period; ++i) ema += prices[i];
    ema /= period;

    // Apply EMA formula for subsequent prices up to 'index'
    double multiplier = 2.0 / (period + 1.0);
    for(size_t i = period; i < prices.size(); ++i) {
        ema = (prices[i] - ema) * multiplier + ema;
    }
    return ema;
}

double TechnicalAnalysis::_calculateRSI(int period, int index) const { // Added const
    if (index < period || period <= 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_ and boundary check
        return 50.0; // Default to neutral if not enough data or invalid period
    }
    
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    
    // Calculate initial average gain and loss
    for (int i = index - period + 1; i <= index; i++) {
        if (i <= 0) return 50.0; // Not enough historical data for comparison
        double change = (*price_series_ptr_)[i].close - (*price_series_ptr_)[i-1].close; // Used price_series_ptr_
        if (change > 0) {
            gain_sum += change;
        } else {
            loss_sum += std::abs(change);
        }
    }
    
    if (period == 0) return 50.0; // Avoid division by zero
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    // Avoid division by zero for RS
    if (avg_loss == 0.0) {
        return 100.0; // If all losses are zero, RSI is 100
    }
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

std::pair<double, double> TechnicalAnalysis::_calculateMACD(int index) const { // Added const
    // MACD typically uses 12-period EMA and 26-period EMA.
    // Signal line is 9-period EMA of MACD line.
    // Need at least 26 periods for MACD line, then 9 more for signal line for full data.
    if (index < 25 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_, need 26 data points (0 to 25)
        return {0.0, 0.0}; // Not enough data
    }
    
    double ema12 = _calculateEMA(12, index);
    double ema26 = _calculateEMA(26, index);
    double macd_line = ema12 - ema26;
    
    // Calculate signal line (9-day EMA of MACD line)
    double signal_line = 0.0;
    int signal_period = 9;
    // Need at least 'signal_period'-1 more data points before 'index' for which MACD can be calculated.
    // So, index must be at least 25 (for ema26) + (signal_period -1) for the start of MACD history.
    if (index >= 25 + signal_period - 1) { 
        std::vector<double> macd_history;
        for (int i = index - signal_period + 1; i <= index; i++) {
            // Ensure 'i' is valid for _calculateEMA calls
            if (i < 25) { // Should not happen if outer condition is correct
                 macd_history.push_back(0); // Or handle error
                 continue;
            }
            double hist_ema12 = _calculateEMA(12, i);
            double hist_ema26 = _calculateEMA(26, i);
            macd_history.push_back(hist_ema12 - hist_ema26);
        }
        
        if (macd_history.size() < static_cast<size_t>(signal_period)) { // Should not happen
            return {macd_line, macd_line}; // Default if somehow not enough history
        }

        // Calculate EMA of MACD history for signal line
        double sum = 0.0;
        for(int i=0; i<signal_period; ++i) sum += macd_history[i];
        signal_line = sum / signal_period; // Initial SMA for EMA

        double multiplier = 2.0 / (signal_period + 1.0);
        for (size_t i = signal_period; i < macd_history.size(); i++) { // This loop won't run if macd_history.size() == signal_period
             // The EMA calculation should iterate from the (period)th element using the (period-1)th EMA
             // Corrected EMA calculation for a series:
        }
        // Correct EMA calculation for the macd_history series
        // Initial EMA (SMA) is already calculated as signal_line
        for (size_t i = 1; i < macd_history.size(); ++i) { // Iterate starting from the second element of the history window
            signal_line = (macd_history[i] - signal_line) * multiplier + signal_line;
        }


    } else {
        signal_line = macd_line; // Default if not enough history for signal line EMA
    }
    
    return {macd_line, signal_line};
}

std::tuple<double, double, double> TechnicalAnalysis::_calculateBollingerBands(int period, int index, double stdDev) const { // Added const, changed return type
    if (index < period - 1 || period <= 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_
        double price = 0.0;
        if (!(*price_series_ptr_).empty() && index >=0 && index < static_cast<int>((*price_series_ptr_).size())) price = (*price_series_ptr_)[index].close;
        return std::make_tuple(price, price, price); // Not enough data
    }
    
    // Calculate SMA (Middle Band)
    double middle = _calculateSMA(period, index);
    
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int i = index - period + 1; i <= index; i++) {
        double diff = (*price_series_ptr_)[i].close - middle; // Used price_series_ptr_
        sum_sq_diff += diff * diff;
    }
    if (period == 0) return std::make_tuple(middle,middle,middle); // Avoid division by zero
    double variance = sum_sq_diff / period;
    double std_deviation = std::sqrt(variance);
    
    // Calculate upper and lower bands
    double upper = middle + (stdDev * std_deviation);
    double lower = middle - (stdDev * std_deviation);
    
    return std::make_tuple(upper, middle, lower); // Corrected return for std::tuple
}

double TechnicalAnalysis::_calculateATR(int period, int index) const { // Added const
    if (index < 1 || period <= 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_
        return 0.0; // Not enough data or invalid period
    }
    
    std::vector<double> true_ranges;
    for (int i = std::max(1, index - period + 1); i <= index; ++i) {
        double high = (*price_series_ptr_)[i].high;
        double low = (*price_series_ptr_)[i].low;
        double prev_close = (*price_series_ptr_)[i-1].close;
        double tr = std::max({high - low, std::abs(high - prev_close), std::abs(low - prev_close)});
        true_ranges.push_back(tr);
    }

    if (true_ranges.empty()) return 0.0;

    // Wilder's smoothing for ATR: ATR = (Previous ATR * (n-1) + Current TR) / n
    // For the first ATR, it's the average of TRs for 'period'
    if (static_cast<int>(true_ranges.size()) < period) { // Not enough TRs for full ATR calculation yet
        // Could return average of available TRs or 0
        return std::accumulate(true_ranges.begin(), true_ranges.end(), 0.0) / true_ranges.size();
    }
    
    // Calculate initial ATR (SMA of TRs for the first 'period' values in our window)
    double atr = 0.0;
    // We need 'period' TR values. The loop for true_ranges collects up to 'period' values ending at 'index'.
    // So true_ranges should contain 'period' values if index >= period-1.
    // The ATR calculation is typically iterative.
    // The recursive call `_calculateATR(period, index - 1)` is one way.
    // Let's stick to the recursive definition provided, assuming it's intended.
    // Base case for recursion: if index < period, calculate simple average of TRs up to index.
    
    double current_tr = std::max({
        (*price_series_ptr_)[index].high - (*price_series_ptr_)[index].low,
        std::abs((*price_series_ptr_)[index].high - (*price_series_ptr_)[index-1].close),
        std::abs((*price_series_ptr_)[index].low - (*price_series_ptr_)[index-1].close)
    });

    if (index < period) { // For the first 'period-1' indices, ATR is just TR or avg of TRs.
                          // Let's use average of TRs for the first 'period' calculations.
        double sum_tr = 0;
        int count = 0;
        for(int k=1; k<=index; ++k){ // Iterate up to current index 'k' (which is 'index' in this context)
             sum_tr += std::max({(*price_series_ptr_)[k].high - (*price_series_ptr_)[k].low,
                                 std::abs((*price_series_ptr_)[k].high - (*price_series_ptr_)[k-1].close),
                                 std::abs((*price_series_ptr_)[k].low - (*price_series_ptr_)[k-1].close)});
            count++;
        }
        if(count == 0) return 0.0;
        return sum_tr/count;
    }
    
    // Recursive step for Wilder's smoothing
    double prev_atr = _calculateATR(period, index - 1);
    return ((prev_atr * (period - 1)) + current_tr) / period;
}

double TechnicalAnalysis::_calculateOBV(int index) const { // Added const
    if (index < 0 || index >= static_cast<int>((*price_series_ptr_).size())) return 0.0; // Boundary check

    if (index == 0) { // Base case for OBV
        return (*price_series_ptr_)[index].volume; // Used price_series_ptr_ (or 0 if preferred for index 0)
    }
    
    double obv = _calculateOBV(index - 1); // Recursive call
    
    if ((*price_series_ptr_)[index].close > (*price_series_ptr_)[index-1].close) { // Used price_series_ptr_
        obv += (*price_series_ptr_)[index].volume; // Used price_series_ptr_
    } else if ((*price_series_ptr_)[index].close < (*price_series_ptr_)[index-1].close) { // Used price_series_ptr_
        obv -= (*price_series_ptr_)[index].volume; // Used price_series_ptr_
    }
    // If close is equal, OBV remains unchanged.
    
    return obv;
}

double TechnicalAnalysis::_calculateStochastic(int period, int index) const { // Added const
    if (index < period - 1 || period <= 0 || index >= static_cast<int>((*price_series_ptr_).size())) { // Used price_series_ptr_
        return 50.0; // Default to middle if not enough data
    }
    
    // Find highest high and lowest low over the period
    double highest_high = (*price_series_ptr_)[index - period + 1].high; // Used price_series_ptr_
    double lowest_low = (*price_series_ptr_)[index - period + 1].low;   // Used price_series_ptr_
    
    for (int i = index - period + 2; i <= index; i++) {
        highest_high = std::max(highest_high, (*price_series_ptr_)[i].high); // Used price_series_ptr_
        lowest_low = std::min(lowest_low, (*price_series_ptr_)[i].low);     // Used price_series_ptr_
    }
    
    // Calculate %K
    double range = highest_high - lowest_low;
    if (range == 0.0) { // Avoid division by zero
        // If range is 0, it means all prices in period were same.
        // If current close is also same, position is undefined. 50.0 is a neutral choice.
        // Or, if current close equals lowest_low (and highest_high), it could be 0 or 100.
        // Let's check if close is also part of this flat line.
        if ((*price_series_ptr_)[index].close == lowest_low) return 0.0; // If close is at the low end of flat line
        if ((*price_series_ptr_)[index].close == highest_high) return 100.0; // If close is at the high end of flat line
        return 50.0; // Otherwise neutral
    }
    
    return (((*price_series_ptr_)[index].close - lowest_low) / range) * 100.0; // Used price_series_ptr_
}

std::vector<double> TechnicalAnalysis::_getPrices(int start, int end) const { // Added const
    std::vector<double> prices;
    // Ensure start and end are within bounds of the actual data
    int actual_start = std::max(0, start);
    int actual_end = std::min(end, static_cast<int>((*price_series_ptr_).size()) - 1); // Used price_series_ptr_

    if (actual_start > actual_end) return prices; // Empty range

    for (int i = actual_start; i <= actual_end; i++) {
        prices.push_back((*price_series_ptr_)[i].close); // Used price_series_ptr_
    }
    return prices;
}

std::vector<double> TechnicalAnalysis::_getVolumes(int start, int end) const { // Added const
    std::vector<double> volumes;
    // Ensure start and end are within bounds
    int actual_start = std::max(0, start);
    int actual_end = std::min(end, static_cast<int>((*price_series_ptr_).size()) - 1); // Used price_series_ptr_

    if (actual_start > actual_end) return volumes; // Empty range

    for (int i = actual_start; i <= actual_end; i++) {
        volumes.push_back((*price_series_ptr_)[i].volume); // Used price_series_ptr_
    }
    return volumes;
}
