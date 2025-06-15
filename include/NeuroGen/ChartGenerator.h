#pragma once
#ifndef CHART_GENERATOR_H
#define CHART_GENERATOR_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <map>
#include <NeuroGen/TradingAgent.h>
#include <NeuroGen/NetworkIntegration.h>

/**
 * @brief Comprehensive chart generation and performance analysis system
 * 
 * The ChartGenerator analyzes trading agent performance, neural network statistics,
 * learning progress, and generates detailed HTML charts with interactive visualizations.
 * It provides insights into trading accuracy, risk management effectiveness, 
 * neural network learning progress, and system performance metrics.
 */
class ChartGenerator {
public:
    /**
     * @brief Data structures for different types of performance metrics
     */
    struct TradingPerformanceData {
        std::vector<float> timestamps;
        std::vector<float> portfolio_values;
        std::vector<float> price_data;
        std::vector<float> realized_pnl;
        std::vector<float> unrealized_pnl;
        std::vector<float> drawdown_values;
        std::vector<std::string> trade_actions;
        std::vector<float> trade_amounts;
        std::vector<float> confidence_levels;
        std::vector<float> cumulative_returns;
        
        // Risk metrics
        std::vector<float> position_sizes;
        std::vector<float> volatility_readings;
        std::vector<float> var_estimates;
        
        // Performance ratios
        float sharpe_ratio;
        float max_drawdown;
        float total_return;
        float win_rate;
        int total_trades;
        float profit_factor;
        float avg_trade_duration_hours;
    };
    
    struct NeuralNetworkData {
        std::vector<float> timestamps;
        std::vector<float> reward_history;
        std::vector<float> confidence_history;
        std::vector<float> learning_rates;
        std::vector<float> average_firing_rates;
        std::vector<float> network_stability;
        std::vector<float> weight_change_magnitudes;
        std::vector<int> total_spikes;
        std::vector<float> network_efficiency;
        std::vector<float> learning_progress;
        
        // Decision analysis
        std::map<std::string, int> action_counts;
        std::vector<float> decision_accuracy;
        std::vector<float> prediction_errors;
        
        // Learning statistics
        float average_reward;
        float reward_variance;
        float learning_convergence_rate;
        float network_adaptation_score;
    };
    
    struct SystemPerformanceData {
        std::vector<float> timestamps;
        std::vector<double> simulation_times;
        std::vector<double> learning_update_times;
        std::vector<double> network_update_times;
        std::vector<int> timesteps_per_second;
        std::vector<float> memory_usage_mb;
        std::vector<float> cpu_utilization;
        std::vector<float> gpu_utilization;
        
        // System health metrics
        double total_simulation_time;
        int total_timesteps;
        int learning_updates;
        float average_fps;
        float system_efficiency;
    };
    
    struct RiskAnalysisData {
        std::vector<float> timestamps;
        std::vector<float> portfolio_risk;
        std::vector<float> position_risk;
        std::vector<float> concentration_risk;
        std::vector<float> volatility_risk;
        std::vector<float> var_breaches;
        std::vector<float> risk_adjusted_returns;
        
        // Risk thresholds and limits
        float max_position_size_limit;
        float max_risk_per_trade_limit;
        float volatility_threshold;
        float confidence_threshold;
        
        // Risk performance metrics
        float risk_efficiency_ratio;
        float maximum_adverse_excursion;
        float maximum_favorable_excursion;
        int risk_violations_count;
    };

private:
    // Data aggregation and analysis
    TradingPerformanceData trading_data_;
    NeuralNetworkData neural_data_;
    SystemPerformanceData system_data_;
    RiskAnalysisData risk_data_;
    
    // Chart generation settings
    struct ChartSettings {
        int width = 1200;
        int height = 800;
        std::string color_scheme = "modern";
        bool interactive = true;
        bool show_grid = true;
        bool show_legend = true;
        int data_point_limit = 10000; // For performance
    } chart_settings_;
    
    // Analysis timeframes
    enum class TimeFrame {
        REAL_TIME,      // Last 100 data points
        SHORT_TERM,     // Last 1000 data points  
        MEDIUM_TERM,    // Last 5000 data points
        LONG_TERM,      // All available data
        CUSTOM          // User-defined range
    };

public:
    /**
     * @brief Constructor
     */
    ChartGenerator();
    
    /**
     * @brief Destructor
     */
    ~ChartGenerator();
    
    /**
     * @brief Main analysis and chart generation methods
     */
    
    // Data collection from system components
    void collectTradingData(const TradingAgent& agent);
    void collectNeuralNetworkData(const EnhancedNetworkManager& network_manager);
    void collectSystemPerformanceData(const EnhancedNetworkManager& network_manager);
    void collectRiskAnalysisData(const TradingAgent& agent);
    
    // Comprehensive analysis generation
    bool generateComprehensiveReport(const std::string& output_directory = "charts/");
    bool generateTradingPerformanceCharts(const std::string& filename);
    bool generateNeuralNetworkAnalysisCharts(const std::string& filename);
    bool generateSystemPerformanceCharts(const std::string& filename);
    bool generateRiskAnalysisCharts(const std::string& filename);
    bool generateLearningProgressCharts(const std::string& filename);
    
    // Real-time dashboard generation
    bool generateRealTimeDashboard(const std::string& filename);
    void updateRealTimeDashboard(const TradingAgent& agent, 
                                const EnhancedNetworkManager& network_manager);
    
    // Specialized analysis charts
    bool generateCorrelationAnalysis(const std::string& filename);
    bool generateVolatilityAnalysis(const std::string& filename);
    bool generateDecisionAccuracyAnalysis(const std::string& filename);
    bool generateLearningEfficiencyAnalysis(const std::string& filename);
    bool generateRiskReturnOptimization(const std::string& filename);
    
    // Interactive analysis tools
    bool generateInteractivePortfolioAnalysis(const std::string& filename);
    bool generateInteractiveNeuralNetworkVisualization(const std::string& filename);
    bool generateInteractiveRiskDashboard(const std::string& filename);
    
    // Statistical analysis and insights
    struct AnalysisInsights {
        // Trading insights
        std::string performance_summary;
        std::vector<std::string> trading_strengths;
        std::vector<std::string> trading_weaknesses;
        std::vector<std::string> improvement_recommendations;
        
        // Neural network insights
        std::string learning_assessment;
        std::vector<std::string> network_strengths;
        std::vector<std::string> network_issues;
        std::vector<std::string> optimization_suggestions;
        
        // Risk management insights
        std::string risk_assessment;
        std::vector<std::string> risk_strengths;
        std::vector<std::string> risk_concerns;
        std::vector<std::string> risk_recommendations;
        
        // System performance insights
        std::string system_assessment;
        std::vector<std::string> performance_highlights;
        std::vector<std::string> bottlenecks;
        std::vector<std::string> optimization_opportunities;
    };
    
    AnalysisInsights generateDetailedInsights();
    bool exportInsightsReport(const AnalysisInsights& insights, const std::string& filename);
    
    // Configuration and customization
    void setChartSettings(const ChartSettings& settings);
    void setTimeFrame(TimeFrame frame, int custom_points = -1);
    void setColorScheme(const std::string& scheme);
    void enableInteractivity(bool enable);
    
    // Data export capabilities
    bool exportTradingDataCSV(const std::string& filename);
    bool exportNeuralNetworkDataCSV(const std::string& filename);
    bool exportSystemPerformanceDataCSV(const std::string& filename);
    bool exportRiskAnalysisDataCSV(const std::string& filename);
    
    // Performance benchmarking
    struct BenchmarkResults {
        float vs_buy_and_hold_return;
        float vs_random_trading_return;
        float vs_simple_moving_average_strategy;
        float information_ratio;
        float tracking_error;
        float alpha;
        float beta;
    };
    
    BenchmarkResults generateBenchmarkComparison(const std::vector<float>& market_prices);
    bool generateBenchmarkCharts(const BenchmarkResults& results, const std::string& filename);

private:
    // Internal chart generation helpers
    std::string generateHTMLChartHeader();
    std::string generateHTMLChartFooter();
    std::string generateChartJS(const std::string& chart_id, const std::string& chart_config);
    std::string generateTimeSeriesChart(const std::vector<float>& timestamps,
                                       const std::vector<float>& values,
                                       const std::string& title,
                                       const std::string& y_label,
                                       const std::string& color = "#3498db");
    std::string generateMultiSeriesChart(const std::vector<float>& timestamps,
                                        const std::vector<std::vector<float>>& series_data,
                                        const std::vector<std::string>& series_labels,
                                        const std::string& title,
                                        const std::string& y_label);
    std::string generateCandlestickChart(const std::vector<float>& timestamps,
                                        const std::vector<float>& open,
                                        const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        const std::string& title);
    std::string generateScatterPlot(const std::vector<float>& x_data,
                                   const std::vector<float>& y_data,
                                   const std::string& title,
                                   const std::string& x_label,
                                   const std::string& y_label);
    std::string generateHistogram(const std::vector<float>& data,
                                 const std::string& title,
                                 int bins = 50);
    std::string generateHeatmap(const std::vector<std::vector<float>>& correlation_matrix,
                               const std::vector<std::string>& labels,
                               const std::string& title);
    
    // Data analysis helpers
    float calculateSharpeRatio(const std::vector<float>& returns, float risk_free_rate = 0.0f);
    float calculateMaxDrawdown(const std::vector<float>& values);
    float calculateVaR(const std::vector<float>& returns, float confidence_level = 0.05f);
    std::vector<float> calculateRollingVolatility(const std::vector<float>& prices, int window = 20);
    std::vector<float> calculateMovingAverage(const std::vector<float>& data, int window);
    std::vector<float> calculateBollingerBands(const std::vector<float>& prices, int window = 20, float std_dev = 2.0f);
    std::vector<std::vector<float>> calculateCorrelationMatrix(const std::vector<std::vector<float>>& data);
    
    // Statistical analysis
    float calculateMean(const std::vector<float>& data);
    float calculateStdDev(const std::vector<float>& data);
    float calculateSkewness(const std::vector<float>& data);
    float calculateKurtosis(const std::vector<float>& data);
    float calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y);
    
    // Performance optimization
    void optimizeDataForVisualization(std::vector<float>& data, int max_points = 1000);
    void aggregateHighFrequencyData(std::vector<float>& timestamps, 
                                   std::vector<float>& values, 
                                   int target_points);
    
    // Utility functions
    std::string formatTimestamp(float timestamp);
    std::string formatCurrency(float value);
    std::string formatPercentage(float value);
    std::string getColorByPerformance(float performance);
    std::string generateUniqueChartId();
    
    // Configuration validation
    bool validateChartSettings();
    bool validateDataIntegrity();
    
    // Error handling and logging
    void logError(const std::string& message);
    void logWarning(const std::string& message);
    void logInfo(const std::string& message);
    
    // Internal state
    TimeFrame current_timeframe_;
    int custom_point_limit_;
    std::string last_error_;
    bool data_collected_;
    std::chrono::system_clock::time_point last_update_;
    
    // Chart ID management for multiple charts in one HTML file
    int chart_counter_;
    std::vector<std::string> generated_chart_ids_;
};

#endif // CHART_GENERATOR_H
