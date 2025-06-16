#ifndef NEUROGEN_TECHNICALANALYSIS_H
#define NEUROGEN_TECHNICALANALYSIS_H

#include <vector>
#include <string>
#include <map>
#include <utility> // For std::pair
#include <tuple>   // For std::tuple
#include "NeuroGen/PriceTick.h" // Will be copied to include/NeuroGen/

class TechnicalAnalysis {
public:
    /**
     * @brief Constructor.
     * @param price_series A reference to the historical price data.
     */
    TechnicalAnalysis(const std::vector<PriceTick>& price_series);

    /**
     * @brief Calculates and returns a map of technical indicators for a given tick index.
     * @param index The index in the price_series for which to calculate features.
     * @return A map where keys are feature names (e.g., "RSI_14") and values are their computed values.
     */
    std::map<std::string, double> getFeaturesForTick(int index);

    /**
     * @brief Updates the price series used for analysis.
     * @param price_series A reference to the new historical price data.
     */
    void updatePriceSeries(const std::vector<PriceTick>& price_series);

    // Public methods for calculating indicators (non-private versions)
    void calculateSMA(const std::vector<PriceTick>& price_series, int period);
    void calculateEMA(const std::vector<PriceTick>& price_series, int period);
    void calculateRSI(const std::vector<PriceTick>& price_series, int period);

private:
    const std::vector<PriceTick>* price_series_ptr_; // Pointer to avoid dangling reference if original vector is temporary

    // Private helper methods for calculating specific indicators
    double _calculateSMA(int period, int index) const;
    double _calculateEMA(int period, int index) const; // Signature changed
    double _calculateRSI(int period, int index) const;
    std::pair<double, double> _calculateMACD(int index) const; // Return type changed
    
    // Added declarations for missing helper methods
    std::tuple<double, double, double> _calculateBollingerBands(int period, int index, double stdDev) const;
    double _calculateATR(int period, int index) const;
    double _calculateOBV(int index) const;
    double _calculateStochastic(int period, int index) const;
    std::vector<double> _getPrices(int start, int end) const;
    std::vector<double> _getVolumes(int start, int end) const;
};

#endif // NEUROGEN_TECHNICALANALYSIS_H
