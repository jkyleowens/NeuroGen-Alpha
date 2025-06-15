// ChartGenerator.cpp - Comprehensive chart generation and analysis for NeuroGen-Alpha trading system
#include <NeuroGen/ChartGenerator.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace NeuroGen {

ChartGenerator::ChartGenerator(const std::string& output_dir)
    : output_directory_(output_dir) {
    
    // Create output directory if it doesn't exist
    std::system(("mkdir -p " + output_directory_).c_str());
    
    std::cout << "[ChartGenerator] Initialized with output directory: " << output_directory_ << std::endl;
}

ChartGenerator::~ChartGenerator() = default;

// === Data Collection Methods ===

TradingPerformanceData ChartGenerator::collectTradingData(const TradingAgent& agent) const {
    TradingPerformanceData data;
    
    try {
        // Get agent statistics
        auto stats = agent.getStatistics();
        
        // Basic performance metrics
        data.total_trades = stats.total_trades;
        data.profitable_trades = stats.profitable_trades;
        data.total_profit_loss = stats.total_profit_loss;
        data.win_rate = (data.total_trades > 0) ? 
            static_cast<float>(data.profitable_trades) / data.total_trades : 0.0f;
        
        // Get portfolio information
        auto portfolio = agent.getPortfolio();
        if (portfolio) {
            data.current_balance = portfolio->getTotalValue();
            data.initial_balance = portfolio->getInitialBalance();
            data.total_return = (data.initial_balance > 0) ? 
                (data.current_balance - data.initial_balance) / data.initial_balance : 0.0f;
        }
        
        // Trading history - get recent decisions
        data.trading_history = agent.getRecentDecisions(100); // Last 100 decisions
        
        // Calculate time-based metrics
        calculateTimeBasedMetrics(data);
        
        // Portfolio evolution
        data.portfolio_evolution = agent.getPortfolioHistory();
        
        std::cout << "[ChartGenerator] Collected trading data: " 
                  << data.total_trades << " trades, " 
                  << (data.win_rate * 100) << "% win rate" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error collecting trading data: " << e.what() << std::endl;
    }
    
    return data;
}

NeuralNetworkData ChartGenerator::collectNeuralNetworkData(const TradingAgent& agent) const {
    NeuralNetworkData data;
    
    try {
        // Get neural network from agent
        auto network = agent.getNeuralNetwork();
        if (!network) {
            std::cerr << "[ChartGenerator] No neural network available" << std::endl;
            return data;
        }
        
        // Get network statistics
        auto network_stats = network->getStatistics();
        
        // Basic network info
        data.total_neurons = network_stats.total_neurons;
        data.total_synapses = network_stats.total_synapses;
        data.average_firing_rate = network_stats.average_firing_rate;
        data.network_activity = network_stats.network_activity;
        
        // Learning metrics
        data.learning_rate = agent.getLearningRate();
        data.total_rewards = agent.getTotalRewards();
        data.recent_rewards = agent.getRewardHistory();
        
        // Calculate average reward
        if (!data.recent_rewards.empty()) {
            data.average_reward = std::accumulate(data.recent_rewards.begin(), 
                                                data.recent_rewards.end(), 0.0f) / data.recent_rewards.size();
        }
        
        // Weight distribution analysis
        analyzeWeightDistribution(network, data);
        
        // Layer-specific analysis
        analyzeLayers(network, data);
        
        std::cout << "[ChartGenerator] Collected neural network data: " 
                  << data.total_neurons << " neurons, " 
                  << data.total_synapses << " synapses" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error collecting neural network data: " << e.what() << std::endl;
    }
    
    return data;
}

SystemPerformanceData ChartGenerator::collectSystemData(const TradingAgent& agent) const {
    SystemPerformanceData data;
    
    try {
        // Get system metrics
        auto start_time = std::chrono::steady_clock::now();
        
        // Memory usage (approximate)
        data.memory_usage_mb = getMemoryUsage();
        
        // CPU usage (if available)
        data.cpu_usage_percent = getCPUUsage();
        
        // GPU metrics (if CUDA is available)
        data.gpu_utilization = getGPUUtilization();
        data.gpu_memory_usage = getGPUMemoryUsage();
        
        // Network processing metrics
        auto network = agent.getNeuralNetwork();
        if (network) {
            auto network_stats = network->getStatistics();
            data.network_update_frequency = network_stats.update_frequency;
            data.average_computation_time = network_stats.average_computation_time;
        }
        
        // Trading system metrics
        data.decision_latency = agent.getAverageDecisionTime();
        data.data_processing_rate = agent.getDataProcessingRate();
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        data.average_computation_time = static_cast<float>(duration.count()) / 1000.0f; // Convert to milliseconds
        
        std::cout << "[ChartGenerator] Collected system performance data" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error collecting system data: " << e.what() << std::endl;
    }
    
    return data;
}

RiskAnalysisData ChartGenerator::collectRiskData(const TradingAgent& agent) const {
    RiskAnalysisData data;
    
    try {
        // Get risk manager
        auto risk_manager = agent.getRiskManager();
        if (!risk_manager) {
            std::cerr << "[ChartGenerator] No risk manager available" << std::endl;
            return data;
        }
        
        // Basic risk metrics
        data.max_drawdown = calculateMaxDrawdown(agent.getPortfolioHistory());
        data.volatility = calculateVolatility(agent.getPortfolioHistory());
        data.sharpe_ratio = calculateSharpeRatio(agent.getPortfolioHistory());
        data.var_95 = calculateVaR(agent.getPortfolioHistory(), 0.95f);
        data.var_99 = calculateVaR(agent.getPortfolioHistory(), 0.99f);
        
        // Risk manager metrics
        data.risk_exposure = risk_manager->getCurrentExposure();
        data.position_size_limits = risk_manager->getPositionSizeLimits();
        data.stop_loss_levels = risk_manager->getStopLossLevels();
        
        // Portfolio composition
        auto portfolio = agent.getPortfolio();
        if (portfolio) {
            data.portfolio_composition = portfolio->getAssetAllocation();
            data.concentration_risk = calculateConcentrationRisk(data.portfolio_composition);
        }
        
        std::cout << "[ChartGenerator] Collected risk analysis data: " 
                  << "Sharpe ratio: " << data.sharpe_ratio 
                  << ", Max drawdown: " << (data.max_drawdown * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error collecting risk data: " << e.what() << std::endl;
    }
    
    return data;
}

// === Chart Generation Methods ===

bool ChartGenerator::generateTradingPerformanceChart(const TradingPerformanceData& data, 
                                                   const std::string& filename) const {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Trading Performance Dashboard");
        file << generateTradingPerformanceHTML(data);
        file << generateHTMLFooter();
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated trading performance chart: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating trading performance chart: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateNeuralNetworkChart(const NeuralNetworkData& data, 
                                              const std::string& filename) const {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Neural Network Analysis Dashboard");
        file << generateNeuralNetworkHTML(data);
        file << generateHTMLFooter();
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated neural network chart: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating neural network chart: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateSystemPerformanceChart(const SystemPerformanceData& data, 
                                                  const std::string& filename) const {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("System Performance Dashboard");
        file << generateSystemPerformanceHTML(data);
        file << generateHTMLFooter();
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated system performance chart: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating system performance chart: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateRiskAnalysisChart(const RiskAnalysisData& data, 
                                             const std::string& filename) const {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Risk Analysis Dashboard");
        file << generateRiskAnalysisHTML(data);
        file << generateHTMLFooter();
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated risk analysis chart: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating risk analysis chart: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateComprehensiveDashboard(const TradingAgent& agent, 
                                                  const std::string& filename) const {
    try {
        // Collect all data
        auto trading_data = collectTradingData(agent);
        auto network_data = collectNeuralNetworkData(agent);
        auto system_data = collectSystemData(agent);
        auto risk_data = collectRiskData(agent);
        
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("NeuroGen-Alpha Comprehensive Dashboard");
        
        // Dashboard overview
        file << generateDashboardOverview(trading_data, network_data, system_data, risk_data);
        
        // Individual sections
        file << generateTradingPerformanceHTML(trading_data);
        file << generateNeuralNetworkHTML(network_data);
        file << generateSystemPerformanceHTML(system_data);
        file << generateRiskAnalysisHTML(risk_data);
        
        file << generateHTMLFooter();
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated comprehensive dashboard: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating comprehensive dashboard: " << e.what() << std::endl;
        return false;
    }
}

// === Statistical Analysis Helpers ===

float ChartGenerator::calculateSharpeRatio(const std::vector<float>& returns, float risk_free_rate) const {
    if (returns.empty()) return 0.0f;
    
    // Calculate mean return
    float mean_return = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (float ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    float std_dev = std::sqrt(variance);
    
    if (std_dev == 0.0f) return 0.0f;
    
    return (mean_return - risk_free_rate) / std_dev;
}

float ChartGenerator::calculateMaxDrawdown(const std::vector<float>& portfolio_values) const {
    if (portfolio_values.empty()) return 0.0f;
    
    float max_drawdown = 0.0f;
    float peak = portfolio_values[0];
    
    for (float value : portfolio_values) {
        if (value > peak) {
            peak = value;
        }
        
        float drawdown = (peak - value) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    return max_drawdown;
}

float ChartGenerator::calculateVolatility(const std::vector<float>& returns) const {
    if (returns.size() < 2) return 0.0f;
    
    float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
    
    float variance = 0.0f;
    for (float ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= (returns.size() - 1);
    
    return std::sqrt(variance);
}

float ChartGenerator::calculateVaR(const std::vector<float>& returns, float confidence_level) const {
    if (returns.empty()) return 0.0f;
    
    std::vector<float> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t index = static_cast<size_t>((1.0f - confidence_level) * sorted_returns.size());
    if (index >= sorted_returns.size()) index = sorted_returns.size() - 1;
    
    return -sorted_returns[index]; // VaR is typically reported as a positive number
}

float ChartGenerator::calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y) const {
    if (x.size() != y.size() || x.empty()) return 0.0f;
    
    float mean_x = std::accumulate(x.begin(), x.end(), 0.0f) / x.size();
    float mean_y = std::accumulate(y.begin(), y.end(), 0.0f) / y.size();
    
    float numerator = 0.0f;
    float sum_sq_x = 0.0f;
    float sum_sq_y = 0.0f;
    
    for (size_t i = 0; i < x.size(); ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    float denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator == 0.0f) return 0.0f;
    
    return numerator / denominator;
}

// === Export Methods ===

bool ChartGenerator::exportToCSV(const TradingPerformanceData& data, const std::string& filename) const {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open CSV file: " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file << "timestamp,action,price,amount,confidence,profit_loss,portfolio_value\n";
        
        // Write trading history
        for (const auto& decision : data.trading_history) {
            file << decision.timestamp << ","
                 << decision.actionToString() << ","
                 << decision.suggested_price << ","
                 << decision.suggested_amount << ","
                 << decision.confidence << ","
                 << decision.profit_loss << ","
                 << decision.portfolio_value << "\n";
        }
        
        file.close();
        
        std::cout << "[ChartGenerator] Exported trading data to CSV: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error exporting to CSV: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateInsightsReport(const TradingAgent& agent, const std::string& filename) const {
    try {
        // Collect all data
        auto trading_data = collectTradingData(agent);
        auto network_data = collectNeuralNetworkData(agent);
        auto system_data = collectSystemData(agent);
        auto risk_data = collectRiskData(agent);
        
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open insights report file: " << filepath << std::endl;
            return false;
        }
        
        // Generate comprehensive insights report
        file << "# NeuroGen-Alpha Trading System Insights Report\n\n";
        file << "Generated on: " << getCurrentTimestamp() << "\n\n";
        
        // Trading Performance Insights
        file << "## Trading Performance Analysis\n\n";
        file << "- Total Trades: " << trading_data.total_trades << "\n";
        file << "- Win Rate: " << std::fixed << std::setprecision(2) << (trading_data.win_rate * 100) << "%\n";
        file << "- Total Return: " << std::fixed << std::setprecision(2) << (trading_data.total_return * 100) << "%\n";
        file << "- Sharpe Ratio: " << std::fixed << std::setprecision(3) << risk_data.sharpe_ratio << "\n";
        file << "- Maximum Drawdown: " << std::fixed << std::setprecision(2) << (risk_data.max_drawdown * 100) << "%\n\n";
        
        // Neural Network Insights
        file << "## Neural Network Analysis\n\n";
        file << "- Network Size: " << network_data.total_neurons << " neurons, " << network_data.total_synapses << " synapses\n";
        file << "- Average Firing Rate: " << std::fixed << std::setprecision(1) << network_data.average_firing_rate << " Hz\n";
        file << "- Learning Rate: " << std::scientific << network_data.learning_rate << "\n";
        file << "- Average Reward: " << std::fixed << std::setprecision(3) << network_data.average_reward << "\n\n";
        
        // System Performance Insights
        file << "## System Performance\n\n";
        file << "- Memory Usage: " << std::fixed << std::setprecision(1) << system_data.memory_usage_mb << " MB\n";
        file << "- CPU Usage: " << std::fixed << std::setprecision(1) << system_data.cpu_usage_percent << "%\n";
        file << "- Average Decision Time: " << std::fixed << std::setprecision(2) << system_data.decision_latency << " ms\n\n";
        
        // Risk Analysis Insights
        file << "## Risk Management\n\n";
        file << "- Volatility: " << std::fixed << std::setprecision(3) << risk_data.volatility << "\n";
        file << "- Value at Risk (95%): " << std::fixed << std::setprecision(2) << (risk_data.var_95 * 100) << "%\n";
        file << "- Current Risk Exposure: " << std::fixed << std::setprecision(2) << (risk_data.risk_exposure * 100) << "%\n\n";
        
        // Recommendations
        file << "## Recommendations\n\n";
        generateRecommendations(file, trading_data, network_data, system_data, risk_data);
        
        file.close();
        
        std::cout << "[ChartGenerator] Generated insights report: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating insights report: " << e.what() << std::endl;
        return false;
    }
}

// === Chart Generation Methods ===

bool ChartGenerator::generateTradingPerformanceCharts(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Trading Performance Analysis");
        
        // Key metrics summary
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (trading_data_.total_return >= 0 ? "positive" : "negative") << "\">";
        file << std::fixed << std::setprecision(2) << (trading_data_.total_return * 100) << "%</div>\n";
        file << "        <div class=\"metric-label\">Total Return</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << trading_data_.total_trades << "</div>\n";
        file << "        <div class=\"metric-label\">Total Trades</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (trading_data_.win_rate >= 0.5 ? "positive" : "negative") << "\">";
        file << std::fixed << std::setprecision(1) << (trading_data_.win_rate * 100) << "%</div>\n";
        file << "        <div class=\"metric-label\">Win Rate</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << trading_data_.sharpe_ratio << "</div>\n";
        file << "        <div class=\"metric-label\">Sharpe Ratio</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Portfolio value chart
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Portfolio Value Over Time</div>\n";
        file << "    <canvas id=\"portfolioValueChart\"></canvas>\n";
        file << "</div>\n";
        
        // Cumulative returns chart
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Cumulative Returns</div>\n";
        file << "    <canvas id=\"cumulativeReturnsChart\"></canvas>\n";
        file << "</div>\n";
        
        // Trading decisions scatter plot
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Trading Decisions by Confidence</div>\n";
        file << "    <canvas id=\"tradingDecisionsChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Portfolio value chart
        file << generateTimeSeriesChartJS("portfolioValueChart", 
                                         trading_data_.timestamps,
                                         trading_data_.portfolio_values,
                                         "Portfolio Value",
                                         "Value ($)");
        
        // Cumulative returns chart
        file << generateTimeSeriesChartJS("cumulativeReturnsChart",
                                         trading_data_.timestamps,
                                         trading_data_.cumulative_returns,
                                         "Cumulative Returns",
                                         "Return (%)");
        
        // Trading decisions chart
        file << generateScatterChartJS("tradingDecisionsChart",
                                      trading_data_.decision_confidence,
                                      trading_data_.decision_profits,
                                      "Trading Decisions",
                                      "Confidence",
                                      "Profit/Loss ($)");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated trading performance charts: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating trading charts: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateNeuralNetworkAnalysisCharts(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Neural Network Analysis");
        
        // Network metrics summary
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << neural_data_.total_neurons << "</div>\n";
        file << "        <div class=\"metric-label\">Total Neurons</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << neural_data_.total_synapses << "</div>\n";
        file << "        <div class=\"metric-label\">Total Synapses</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << neural_data_.average_firing_rate << " Hz</div>\n";
        file << "        <div class=\"metric-label\">Avg Firing Rate</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << neural_data_.average_reward << "</div>\n";
        file << "        <div class=\"metric-label\">Avg Reward</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Learning progress chart
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Learning Progress Over Time</div>\n";
        file << "    <canvas id=\"learningProgressChart\"></canvas>\n";
        file << "</div>\n";
        
        // Network activity chart
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Network Activity</div>\n";
        file << "    <canvas id=\"networkActivityChart\"></canvas>\n";
        file << "</div>\n";
        
        // Neurotransmitter levels chart
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Neurotransmitter Levels</div>\n";
        file << "    <canvas id=\"neurotransmitterChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Learning progress chart
        file << generateTimeSeriesChartJS("learningProgressChart",
                                         neural_data_.timestamps,
                                         neural_data_.learning_progress,
                                         "Learning Progress",
                                         "Accuracy Score");
        
        // Network activity chart
        file << generateMultiSeriesChartJS("networkActivityChart",
                                          neural_data_.timestamps,
                                          {neural_data_.firing_rates, neural_data_.synapse_strengths},
                                          {"Firing Rate", "Synapse Strength"},
                                          "Network Activity",
                                          "Activity Level");
        
        // Neurotransmitter levels chart
        std::vector<std::vector<float>> neurotransmitter_data = {
            neural_data_.dopamine_levels,
            neural_data_.serotonin_levels,
            neural_data_.acetylcholine_levels,
            neural_data_.norepinephrine_levels
        };
        
        file << generateMultiSeriesChartJS("neurotransmitterChart",
                                          neural_data_.timestamps,
                                          neurotransmitter_data,
                                          {"Dopamine", "Serotonin", "Acetylcholine", "Norepinephrine"},
                                          "Neurotransmitter Levels",
                                          "Concentration");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated neural network analysis charts: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating neural network charts: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateSystemPerformanceCharts(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("System Performance Analysis");
        
        // System metrics summary
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << system_data_.average_fps << "</div>\n";
        file << "        <div class=\"metric-label\">Average FPS</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << system_data_.memory_usage_mb.back() << " MB</div>\n";
        file << "        <div class=\"metric-label\">Current Memory Usage</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << system_data_.cpu_utilization.back() << "%</div>\n";
        file << "        <div class=\"metric-label\">Current CPU Usage</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(2) << system_data_.system_efficiency << "%</div>\n";
        file << "        <div class=\"metric-label\">System Efficiency</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Performance monitoring charts
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Resource Utilization Over Time</div>\n";
        file << "    <canvas id=\"resourceUtilizationChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Decision Latency</div>\n";
        file << "    <canvas id=\"decisionLatencyChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Memory Usage</div>\n";
        file << "    <canvas id=\"memoryUsageChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Resource utilization chart
        std::vector<std::vector<float>> resource_data = {
            system_data_.cpu_utilization,
            system_data_.gpu_utilization
        };
        
        file << generateMultiSeriesChartJS("resourceUtilizationChart",
                                          system_data_.timestamps,
                                          resource_data,
                                          {"CPU Utilization", "GPU Utilization"},
                                          "Resource Utilization",
                                          "Utilization (%)");
        
        // Decision latency chart
        file << generateTimeSeriesChartJS("decisionLatencyChart",
                                         system_data_.timestamps,
                                         system_data_.decision_latencies,
                                         "Decision Latency",
                                         "Latency (ms)");
        
        // Memory usage chart
        file << generateTimeSeriesChartJS("memoryUsageChart",
                                         system_data_.timestamps,
                                         system_data_.memory_usage_mb,
                                         "Memory Usage",
                                         "Memory (MB)");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated system performance charts: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating system performance charts: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateRiskAnalysisCharts(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Risk Analysis");
        
        // Risk metrics summary
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (risk_data_.sharpe_ratio >= 1.0 ? "positive" : "negative") << "\">";
        file << std::fixed << std::setprecision(3) << risk_data_.sharpe_ratio << "</div>\n";
        file << "        <div class=\"metric-label\">Sharpe Ratio</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value negative\">" << std::fixed << std::setprecision(2) << (risk_data_.max_drawdown * 100) << "%</div>\n";
        file << "        <div class=\"metric-label\">Max Drawdown</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << risk_data_.volatility << "</div>\n";
        file << "        <div class=\"metric-label\">Volatility</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value negative\">" << std::fixed << std::setprecision(2) << (risk_data_.var_95 * 100) << "%</div>\n";
        file << "        <div class=\"metric-label\">VaR (95%)</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Risk analysis charts
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Risk Exposure Over Time</div>\n";
        file << "    <canvas id=\"riskExposureChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Volatility Analysis</div>\n";
        file << "    <canvas id=\"volatilityChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Risk-Adjusted Returns</div>\n";
        file << "    <canvas id=\"riskAdjustedReturnsChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Risk exposure chart
        file << generateTimeSeriesChartJS("riskExposureChart",
                                         risk_data_.timestamps,
                                         risk_data_.risk_exposure,
                                         "Risk Exposure",
                                         "Exposure Level");
        
        // Volatility chart
        file << generateTimeSeriesChartJS("volatilityChart",
                                         risk_data_.timestamps,
                                         risk_data_.rolling_volatility,
                                         "Rolling Volatility",
                                         "Volatility");
        
        // Risk-adjusted returns chart
        file << generateTimeSeriesChartJS("riskAdjustedReturnsChart",
                                         risk_data_.timestamps,
                                         risk_data_.risk_adjusted_returns,
                                         "Risk-Adjusted Returns",
                                         "Return");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated risk analysis charts: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating risk analysis charts: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateRealTimeDashboard(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open dashboard file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("NeuroGen-Alpha Real-Time Dashboard");
        
        // Dashboard overview metrics
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (trading_data_.portfolio_values.back() > trading_data_.initial_portfolio_value ? "positive" : "negative") << "\">";
        file << formatCurrency(trading_data_.portfolio_values.back()) << "</div>\n";
        file << "        <div class=\"metric-label\">Current Portfolio Value</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (trading_data_.total_return >= 0 ? "positive" : "negative") << "\">";
        file << formatPercentage(trading_data_.total_return) << "</div>\n";
        file << "        <div class=\"metric-label\">Total Return</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << neural_data_.total_neurons << "</div>\n";
        file << "        <div class=\"metric-label\">Active Neurons</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(1) << system_data_.average_fps << "</div>\n";
        file << "        <div class=\"metric-label\">System FPS</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (trading_data_.win_rate >= 0.5 ? "positive" : "negative") << "\">";
        file << formatPercentage(trading_data_.win_rate) << "</div>\n";
        file << "        <div class=\"metric-label\">Win Rate</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << risk_data_.sharpe_ratio << "</div>\n";
        file << "        <div class=\"metric-label\">Sharpe Ratio</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Main dashboard charts
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Portfolio Performance</div>\n";
        file << "    <canvas id=\"dashboardPortfolioChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Neural Network Activity</div>\n";
        file << "    <canvas id=\"dashboardNetworkChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">System Performance</div>\n";
        file << "    <canvas id=\"dashboardSystemChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Risk Monitoring</div>\n";
        file << "    <canvas id=\"dashboardRiskChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts with real-time update capability
        file << "<script>\n";
        
        // Portfolio performance chart
        file << generateTimeSeriesChartJS("dashboardPortfolioChart",
                                         trading_data_.timestamps,
                                         trading_data_.portfolio_values,
                                         "Portfolio Value",
                                         "Value ($)",
                                         "#27ae60");
        
        // Neural network activity chart
        file << generateTimeSeriesChartJS("dashboardNetworkChart",
                                         neural_data_.timestamps,
                                         neural_data_.firing_rates,
                                         "Network Firing Rate",
                                         "Rate (Hz)",
                                         "#3498db");
        
        // System performance chart
        std::vector<std::vector<float>> system_metrics = {
            system_data_.cpu_utilization,
            system_data_.memory_usage_mb
        };
        
        file << generateMultiSeriesChartJS("dashboardSystemChart",
                                          system_data_.timestamps,
                                          system_metrics,
                                          {"CPU %", "Memory MB"},
                                          "System Resources",
                                          "Usage");
        
        // Risk monitoring chart
        file << generateTimeSeriesChartJS("dashboardRiskChart",
                                         risk_data_.timestamps,
                                         risk_data_.risk_exposure,
                                         "Risk Exposure",
                                         "Risk Level",
                                         "#e74c3c");
        
        // Add auto-refresh functionality
        file << R"(
        // Auto-refresh dashboard data
        function refreshDashboard() {
            // This would connect to a real-time data endpoint
            console.log('Refreshing dashboard...');
            setTimeout(refreshDashboard, 5000); // Refresh every 5 seconds
        }
        
        // Start auto-refresh
        setTimeout(refreshDashboard, 5000);
        )";
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated real-time dashboard: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating dashboard: " << e.what() << std::endl;
        return false;
    }
}

// === Chart.js Generation Helpers ===

std::string ChartGenerator::generateTimeSeriesChartJS(const std::string& chart_id,
                                                       const std::vector<float>& timestamps,
                                                       const std::vector<float>& values,
                                                       const std::string& title,
                                                       const std::string& y_label,
                                                       const std::string& color) const {
    std::stringstream js;
    
    js << "const " << chart_id << "Data = {\n";
    js << "    labels: [";
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (i > 0) js << ", ";
        js << "'" << formatTimestamp(timestamps[i]) << "'";
    }
    js << "],\n";
    
    js << "    datasets: [{\n";
    js << "        label: '" << title << "',\n";
    js << "        data: [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) js << ", ";
        js << values[i];
    }
    js << "],\n";
    js << "        borderColor: '" << color << "',\n";
    js << "        backgroundColor: '" << color << "20',\n";
    js << "        borderWidth: 2,\n";
    js << "        fill: false,\n";
    js << "        tension: 0.1\n";
    js << "    }]\n";
    js << "};\n\n";
    
    js << "const " << chart_id << "Config = {\n";
    js << "    type: 'line',\n";
    js << "    data: " << chart_id << "Data,\n";
    js << "    options: {\n";
    js << "        responsive: true,\n";
    js << "        maintainAspectRatio: false,\n";
    js << "        scales: {\n";
    js << "            y: {\n";
    js << "                beginAtZero: false,\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: '" << y_label << "'\n";
    js << "                }\n";
    js << "            },\n";
    js << "            x: {\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: 'Time'\n";
    js << "                }\n";
    js << "            }\n";
    js << "        }\n";
    js << "    }\n";
    js << "};\n\n";
    
    js << "new Chart(document.getElementById('" << chart_id << "'), " << chart_id << "Config);\n\n";
    
    return js.str();
}

std::string ChartGenerator::generateMultiSeriesChartJS(const std::string& chart_id,
                                                        const std::vector<float>& timestamps,
                                                        const std::vector<std::vector<float>>& series_data,
                                                        const std::vector<std::string>& series_labels,
                                                        const std::string& title,
                                                        const std::string& y_label) const {
    std::stringstream js;
    
    const std::vector<std::string> colors = {
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#34495e"
    };
    
    js << "const " << chart_id << "Data = {\n";
    js << "    labels: [";
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (i > 0) js << ", ";
        js << "'" << formatTimestamp(timestamps[i]) << "'";
    }
    js << "],\n";
    
    js << "    datasets: [\n";
    for (size_t series = 0; series < series_data.size(); ++series) {
        if (series > 0) js << ",\n";
        
        std::string color = colors[series % colors.size()];
        
        js << "        {\n";
        js << "            label: '" << series_labels[series] << "',\n";
        js << "            data: [";
        for (size_t i = 0; i < series_data[series].size(); ++i) {
            if (i > 0) js << ", ";
            js << series_data[series][i];
        }
        js << "],\n";
        js << "            borderColor: '" << color << "',\n";
        js << "            backgroundColor: '" << color << "20',\n";
        js << "            borderWidth: 2,\n";
        js << "            fill: false,\n";
        js << "            tension: 0.1\n";
        js << "        }";
    }
    js << "\n    ]\n";
    js << "};\n\n";
    
    js << "const " << chart_id << "Config = {\n";
    js << "    type: 'line',\n";
    js << "    data: " << chart_id << "Data,\n";
    js << "    options: {\n";
    js << "        responsive: true,\n";
    js << "        maintainAspectRatio: false,\n";
    js << "        scales: {\n";
    js << "            y: {\n";
    js << "                beginAtZero: false,\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: '" << y_label << "'\n";
    js << "                }\n";
    js << "            },\n";
    js << "            x: {\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: 'Time'\n";
    js << "                }\n";
    js << "            }\n";
    js << "        }\n";
    js << "    }\n";
    js << "};\n\n";
    
    js << "new Chart(document.getElementById('" << chart_id << "'), " << chart_id << "Config);\n\n";
    
    return js.str();
}

std::string ChartGenerator::generateScatterChartJS(const std::string& chart_id,
                                                    const std::vector<float>& x_data,
                                                    const std::vector<float>& y_data,
                                                    const std::string& title,
                                                    const std::string& x_label,
                                                    const std::string& y_label) const {
    std::stringstream js;
    
    js << "const " << chart_id << "Data = {\n";
    js << "    datasets: [{\n";
    js << "        label: '" << title << "',\n";
    js << "        data: [";
    
    for (size_t i = 0; i < std::min(x_data.size(), y_data.size()); ++i) {
        if (i > 0) js << ", ";
        js << "{x: " << x_data[i] << ", y: " << y_data[i] << "}";
    }
    
    js << "],\n";
    js << "        backgroundColor: '#3498db80',\n";
    js << "        borderColor: '#3498db',\n";
    js << "        borderWidth: 1\n";
    js << "    }]\n";
    js << "};\n\n";
    
    js << "const " << chart_id << "Config = {\n";
    js << "    type: 'scatter',\n";
    js << "    data: " << chart_id << "Data,\n";
    js << "    options: {\n";
    js << "        responsive: true,\n";
    js << "        maintainAspectRatio: false,\n";
    js << "        scales: {\n";
    js << "            x: {\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: '" << x_label << "'\n";
    js << "                }\n";
    js << "            },\n";
    js << "            y: {\n";
    js << "                title: {\n";
    js << "                    display: true,\n";
    js << "                    text: '" << y_label << "'\n";
    js << "                }\n";
    js << "            }\n";
    js << "        }\n";
    js << "    }\n";
    js << "};\n\n";
    
    js << "new Chart(document.getElementById('" << chart_id << "'), " << chart_id << "Config);\n\n";
    
    return js.str();
}

// === Utility Functions ===

std::string ChartGenerator::formatTimestamp(float timestamp) const {
    auto time_point = std::chrono::system_clock::from_time_t(static_cast<time_t>(timestamp));
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    return ss.str();
}

std::string ChartGenerator::formatCurrency(float value) const {
    std::stringstream ss;
    ss << "$" << std::fixed << std::setprecision(2) << value;
    return ss.str();
}

std::string ChartGenerator::formatPercentage(float value) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << (value * 100) << "%";
    return ss.str();
}

void ChartGenerator::generateRecommendations(std::ofstream& file,
                                           const TradingPerformanceData& trading_data,
                                           const NeuralNetworkData& network_data,
                                           const SystemPerformanceData& system_data,
                                           const RiskAnalysisData& risk_data) const {
    
    // Performance-based recommendations
    if (trading_data.win_rate < 0.5) {
        file << "- **Improve Trading Strategy**: Win rate of " << std::fixed << std::setprecision(1) << (trading_data.win_rate * 100) 
             << "% is below 50%. Consider adjusting neural network parameters or training data.\n";
    }
    
    if (risk_data.sharpe_ratio < 1.0) {
        file << "- **Enhance Risk-Adjusted Returns**: Sharpe ratio of " << std::fixed << std::setprecision(3) << risk_data.sharpe_ratio 
             << " suggests room for improvement in risk-adjusted performance.\n";
    }
    
    if (risk_data.max_drawdown > 0.2) {
        file << "- **Reduce Maximum Drawdown**: Current max drawdown of " << std::fixed << std::setprecision(1) << (risk_data.max_drawdown * 100) 
             << "% is high. Consider implementing stricter stop-loss mechanisms.\n";
    }
    
    // Neural network recommendations
    if (network_data.average_firing_rate < 10.0) {
        file << "- **Increase Network Activity**: Average firing rate of " << std::fixed << std::setprecision(1) << network_data.average_firing_rate 
             << " Hz may be too low. Consider adjusting neuron thresholds.\n";
    }
    
    if (network_data.learning_rate < 0.001) {
        file << "- **Optimize Learning Rate**: Current learning rate may be too conservative for rapid adaptation.\n";
    }
    
    // System performance recommendations
    if (system_data.average_fps < 30.0) {
        file << "- **Improve System Performance**: Average FPS of " << std::fixed << std::setprecision(1) << system_data.average_fps 
             << " may impact real-time decision making. Consider optimizing computational load.\n";
    }
    
    if (!system_data.memory_usage_mb.empty() && system_data.memory_usage_mb.back() > 8000) {
        file << "- **Optimize Memory Usage**: High memory usage detected. Consider memory optimization strategies.\n";
    }
    
    // General recommendations
    file << "- **Continue Monitoring**: Regular analysis of these metrics will help identify trends and optimization opportunities.\n";
    file << "- **Diversification**: Consider expanding trading strategies to different market conditions.\n";
    file << "- **Backtesting**: Validate improvements through comprehensive backtesting before live deployment.\n";
}

// === Additional Analysis Methods ===

bool ChartGenerator::generateCorrelationAnalysis(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Correlation Analysis");
        
        // Calculate correlations between different metrics
        float portfolio_network_corr = calculateCorrelation(trading_data_.portfolio_values, neural_data_.firing_rates);
        float performance_confidence_corr = calculateCorrelation(trading_data_.decision_profits, trading_data_.decision_confidence);
        float risk_return_corr = calculateCorrelation(risk_data_.risk_exposure, trading_data_.cumulative_returns);
        
        // Correlation metrics summary
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (portfolio_network_corr >= 0.5 ? "positive" : (portfolio_network_corr <= -0.5 ? "negative" : "neutral")) << "\">";
        file << std::fixed << std::setprecision(3) << portfolio_network_corr << "</div>\n";
        file << "        <div class=\"metric-label\">Portfolio-Network Correlation</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (performance_confidence_corr >= 0.5 ? "positive" : (performance_confidence_corr <= -0.5 ? "negative" : "neutral")) << "\">";
        file << std::fixed << std::setprecision(3) << performance_confidence_corr << "</div>\n";
        file << "        <div class=\"metric-label\">Performance-Confidence Correlation</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value " << (risk_return_corr >= 0.5 ? "positive" : (risk_return_corr <= -0.5 ? "negative" : "neutral")) << "\">";
        file << std::fixed << std::setprecision(3) << risk_return_corr << "</div>\n";
        file << "        <div class=\"metric-label\">Risk-Return Correlation</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Correlation scatter plots
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Portfolio Value vs Network Activity</div>\n";
        file << "    <canvas id=\"portfolioNetworkCorr\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Decision Confidence vs Profit/Loss</div>\n";
        file << "    <canvas id=\"confidenceProfitCorr\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Portfolio-Network correlation scatter plot
        file << generateScatterChartJS("portfolioNetworkCorr",
                                      neural_data_.firing_rates,
                                      trading_data_.portfolio_values,
                                      "Portfolio vs Network Activity",
                                      "Network Firing Rate (Hz)",
                                      "Portfolio Value ($)");
        
        // Confidence-Profit correlation scatter plot
        file << generateScatterChartJS("confidenceProfitCorr",
                                      trading_data_.decision_confidence,
                                      trading_data_.decision_profits,
                                      "Confidence vs Profit",
                                      "Decision Confidence",
                                      "Profit/Loss ($)");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated correlation analysis: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating correlation analysis: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateVolatilityAnalysis(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("Volatility Analysis");
        
        // Calculate rolling volatility
        std::vector<float> rolling_vol = calculateRollingVolatility(trading_data_.portfolio_values, 20);
        
        // Volatility metrics
        float current_volatility = rolling_vol.empty() ? 0.0f : rolling_vol.back();
        float avg_volatility = calculateMean(rolling_vol);
        float vol_of_vol = calculateStdDev(rolling_vol);
        
        file << "<div class=\"metrics-grid\">\n";
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << current_volatility << "</div>\n";
        file << "        <div class=\"metric-label\">Current Volatility</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << avg_volatility << "</div>\n";
        file << "        <div class=\"metric-label\">Average Volatility</div>\n";
        file << "    </div>\n";
        
        file << "    <div class=\"metric-card\">\n";
        file << "        <div class=\"metric-value\">" << std::fixed << std::setprecision(3) << vol_of_vol << "</div>\n";
        file << "        <div class=\"metric-label\">Volatility of Volatility</div>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Volatility charts
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Rolling Volatility (20-period)</div>\n";
        file << "    <canvas id=\"rollingVolatilityChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Return Distribution</div>\n";
        file << "    <canvas id=\"returnDistributionChart\"></canvas>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Rolling volatility chart
        std::vector<float> vol_timestamps(trading_data_.timestamps.begin() + 20, trading_data_.timestamps.end());
        file << generateTimeSeriesChartJS("rollingVolatilityChart",
                                         vol_timestamps,
                                         rolling_vol,
                                         "Rolling Volatility",
                                         "Volatility",
                                         "#f39c12");
        
        // Return distribution histogram (simplified as line chart for Chart.js compatibility)
        file << generateTimeSeriesChartJS("returnDistributionChart",
                                         trading_data_.timestamps,
                                         trading_data_.returns,
                                         "Returns Distribution",
                                         "Return",
                                         "#9b59b6");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated volatility analysis: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating volatility analysis: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateComprehensiveReport(const std::string& output_directory) {
    try {
        // Update output directory
        output_directory_ = output_directory;
        
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_directory_);
        
        bool success = true;
        
        // Generate all individual chart files
        success &= generateTradingPerformanceCharts("trading_performance.html");
        success &= generateNeuralNetworkAnalysisCharts("neural_network_analysis.html");
        success &= generateSystemPerformanceCharts("system_performance.html");
        success &= generateRiskAnalysisCharts("risk_analysis.html");
        success &= generateCorrelationAnalysis("correlation_analysis.html");
        success &= generateVolatilityAnalysis("volatility_analysis.html");
        success &= generateRealTimeDashboard("dashboard.html");
        
        // Generate comprehensive overview
        success &= generateOverviewReport("comprehensive_report.html");
        
        // Export data files
        success &= exportTradingDataCSV("trading_data.csv");
        success &= exportNeuralNetworkDataCSV("neural_network_data.csv");
        success &= exportSystemPerformanceDataCSV("system_performance_data.csv");
        success &= exportRiskAnalysisDataCSV("risk_analysis_data.csv");
        
        if (success) {
            std::cout << "[ChartGenerator] Comprehensive report generated successfully in: " << output_directory_ << std::endl;
        } else {
            std::cerr << "[ChartGenerator] Some components of the comprehensive report failed to generate" << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating comprehensive report: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::generateOverviewReport(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open overview file: " << filepath << std::endl;
            return false;
        }
        
        file << generateHTMLHeader("NeuroGen-Alpha Comprehensive Analysis Report");
        
        // Executive summary
        file << "<div style=\"background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;\">\n";
        file << "    <h2 style=\"color: #2c3e50; margin-bottom: 20px;\">Executive Summary</h2>\n";
        file << "    <div style=\"display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;\">\n";
        
        // Key performance indicators
        file << "        <div style=\"text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;\">\n";
        file << "            <div style=\"font-size: 1.5em; font-weight: bold; color: " << (trading_data_.total_return >= 0 ? "#27ae60" : "#e74c3c") << ";\">";
        file << formatPercentage(trading_data_.total_return) << "</div>\n";
        file << "            <div style=\"color: #7f8c8d;\">Total Return</div>\n";
        file << "        </div>\n";
        
        file << "        <div style=\"text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;\">\n";
        file << "            <div style=\"font-size: 1.5em; font-weight: bold; color: #3498db;\">" << trading_data_.total_trades << "</div>\n";
        file << "            <div style=\"color: #7f8c8d;\">Total Trades</div>\n";
        file << "        </div>\n";
        
        file << "        <div style=\"text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;\">\n";
        file << "            <div style=\"font-size: 1.5em; font-weight: bold; color: " << (trading_data_.win_rate >= 0.5 ? "#27ae60" : "#e74c3c") << ";\">";
        file << formatPercentage(trading_data_.win_rate) << "</div>\n";
        file << "            <div style=\"color: #7f8c8d;\">Win Rate</div>\n";
        file << "        </div>\n";
        
        file << "        <div style=\"text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px;\">\n";
        file << "            <div style=\"font-size: 1.5em; font-weight: bold; color: #9b59b6;\">" << std::fixed << std::setprecision(3) << risk_data_.sharpe_ratio << "</div>\n";
        file << "            <div style=\"color: #7f8c8d;\">Sharpe Ratio</div>\n";
        file << "        </div>\n";
        
        file << "    </div>\n";
        file << "</div>\n";
        
        // Navigation menu
        file << "<div style=\"background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;\">\n";
        file << "    <h3 style=\"color: #2c3e50; margin-bottom: 15px;\">Detailed Analysis Reports</h3>\n";
        file << "    <div style=\"display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;\">\n";
        file << "        <a href=\"trading_performance.html\" style=\"display: block; padding: 15px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Trading Performance</a>\n";
        file << "        <a href=\"neural_network_analysis.html\" style=\"display: block; padding: 15px; background: #e74c3c; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Neural Network Analysis</a>\n";
        file << "        <a href=\"system_performance.html\" style=\"display: block; padding: 15px; background: #2ecc71; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">System Performance</a>\n";
        file << "        <a href=\"risk_analysis.html\" style=\"display: block; padding: 15px; background: #f39c12; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Risk Analysis</a>\n";
        file << "        <a href=\"correlation_analysis.html\" style=\"display: block; padding: 15px; background: #9b59b6; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Correlation Analysis</a>\n";
        file << "        <a href=\"volatility_analysis.html\" style=\"display: block; padding: 15px; background: #1abc9c; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Volatility Analysis</a>\n";
        file << "        <a href=\"dashboard.html\" style=\"display: block; padding: 15px; background: #34495e; color: white; text-decoration: none; border-radius: 5px; text-align: center;\">Real-Time Dashboard</a>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Summary charts in overview
        file << "<div class=\"chart-container\">\n";
        file << "    <div class=\"chart-title\">Portfolio Performance Overview</div>\n";
        file << "    <canvas id=\"overviewPortfolioChart\"></canvas>\n";
        file << "</div>\n";
        
        file << "<div style=\"display: grid; grid-template-columns: 1fr 1fr; gap: 20px;\">\n";
        file << "    <div class=\"chart-container\">\n";
        file << "        <div class=\"chart-title\">Risk vs Return</div>\n";
        file << "        <canvas id=\"riskReturnChart\"></canvas>\n";
        file << "    </div>\n";
        file << "    <div class=\"chart-container\">\n";
        file << "        <div class=\"chart-title\">System Efficiency</div>\n";
        file << "        <canvas id=\"systemEfficiencyChart\"></canvas>\n";
        file << "    </div>\n";
        file << "</div>\n";
        
        // Generate Chart.js scripts
        file << "<script>\n";
        
        // Portfolio overview chart
        file << generateTimeSeriesChartJS("overviewPortfolioChart",
                                         trading_data_.timestamps,
                                         trading_data_.portfolio_values,
                                         "Portfolio Value",
                                         "Value ($)",
                                         "#3498db");
        
        // Risk vs Return scatter plot
        file << generateScatterChartJS("riskReturnChart",
                                      risk_data_.rolling_volatility,
                                      trading_data_.returns,
                                      "Risk vs Return",
                                      "Risk (Volatility)",
                                      "Return");
        
        // System efficiency chart
        std::vector<float> efficiency_data;
        for (size_t i = 0; i < system_data_.timestamps.size(); ++i) {
            efficiency_data.push_back(system_data_.system_efficiency);
        }
        
        file << generateTimeSeriesChartJS("systemEfficiencyChart",
                                         system_data_.timestamps,
                                         efficiency_data,
                                         "System Efficiency",
                                         "Efficiency (%)",
                                         "#2ecc71");
        
        file << "</script>\n";
        file << generateHTMLFooter();
        
        file.close();
        std::cout << "[ChartGenerator] Generated overview report: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error generating overview report: " << e.what() << std::endl;
        return false;
    }
}

// === Data Export Methods ===

bool ChartGenerator::exportTradingDataCSV(const std::string& filename) {
    return exportToCSV(trading_data_, filename);
}

bool ChartGenerator::exportNeuralNetworkDataCSV(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open CSV file: " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file << "timestamp,total_neurons,total_synapses,firing_rate,learning_progress,dopamine,serotonin,acetylcholine,norepinephrine\n";
        
        // Write data
        for (size_t i = 0; i < neural_data_.timestamps.size(); ++i) {
            file << neural_data_.timestamps[i] << ","
                 << neural_data_.total_neurons << ","
                 << neural_data_.total_synapses << ",";
            
            if (i < neural_data_.firing_rates.size()) file << neural_data_.firing_rates[i];
            file << ",";
            if (i < neural_data_.learning_progress.size()) file << neural_data_.learning_progress[i];
            file << ",";
            if (i < neural_data_.dopamine_levels.size()) file << neural_data_.dopamine_levels[i];
            file << ",";
            if (i < neural_data_.serotonin_levels.size()) file << neural_data_.serotonin_levels[i];
            file << ",";
            if (i < neural_data_.acetylcholine_levels.size()) file << neural_data_.acetylcholine_levels[i];
            file << ",";
            if (i < neural_data_.norepinephrine_levels.size()) file << neural_data_.norepinephrine_levels[i];
            file << "\n";
        }
        
        file.close();
        std::cout << "[ChartGenerator] Exported neural network data to CSV: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error exporting neural network data to CSV: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::exportSystemPerformanceDataCSV(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open CSV file: " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file << "timestamp,decision_latency,memory_usage_mb,cpu_utilization,gpu_utilization,fps\n";
        
        // Write data
        for (size_t i = 0; i < system_data_.timestamps.size(); ++i) {
            file << system_data_.timestamps[i] << ",";
            
            if (i < system_data_.decision_latencies.size()) file << system_data_.decision_latencies[i];
            file << ",";
            if (i < system_data_.memory_usage_mb.size()) file << system_data_.memory_usage_mb[i];
            file << ",";
            if (i < system_data_.cpu_utilization.size()) file << system_data_.cpu_utilization[i];
            file << ",";
            if (i < system_data_.gpu_utilization.size()) file << system_data_.gpu_utilization[i];
            file << ",";
            file << system_data_.average_fps;
            file << "\n";
        }
        
        file.close();
        std::cout << "[ChartGenerator] Exported system performance data to CSV: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error exporting system performance data to CSV: " << e.what() << std::endl;
        return false;
    }
}

bool ChartGenerator::exportRiskAnalysisDataCSV(const std::string& filename) {
    try {
        std::string filepath = output_directory_ + "/" + filename;
        std::ofstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "[ChartGenerator] Cannot open CSV file: " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file << "timestamp,risk_exposure,rolling_volatility,risk_adjusted_return,sharpe_ratio,max_drawdown,var_95\n";
        
        // Write data
        for (size_t i = 0; i < risk_data_.timestamps.size(); ++i) {
            file << risk_data_.timestamps[i] << ",";
            
            if (i < risk_data_.risk_exposure.size()) file << risk_data_.risk_exposure[i];
            file << ",";
            if (i < risk_data_.rolling_volatility.size()) file << risk_data_.rolling_volatility[i];
            file << ",";
            if (i < risk_data_.risk_adjusted_returns.size()) file << risk_data_.risk_adjusted_returns[i];
            file << ",";
            file << risk_data_.sharpe_ratio << ",";
            file << risk_data_.max_drawdown << ",";
            file << risk_data_.var_95;
            file << "\n";
        }
        
        file.close();
        std::cout << "[ChartGenerator] Exported risk analysis data to CSV: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ChartGenerator] Error exporting risk analysis data to CSV: " << e.what() << std::endl;
        return false;
    }
}

// === Additional Utility Methods ===

std::vector<float> ChartGenerator::calculateRollingVolatility(const std::vector<float>& prices, int window) const {
    std::vector<float> volatility;
    
    if (prices.size() < static_cast<size_t>(window)) {
        return volatility;
    }
    
    for (size_t i = window; i < prices.size(); ++i) {
        std::vector<float> window_returns;
        
        for (int j = 1; j < window; ++j) {
            size_t idx = i - window + j;
            if (idx > 0 && prices[idx - 1] != 0) {
                float return_val = (prices[idx] - prices[idx - 1]) / prices[idx - 1];
                window_returns.push_back(return_val);
            }
        }
        
        float vol = calculateStdDev(window_returns);
        volatility.push_back(vol);
    }
    
    return volatility;
}

std::vector<float> ChartGenerator::calculateMovingAverage(const std::vector<float>& data, int window) const {
    std::vector<float> ma;
    
    if (data.size() < static_cast<size_t>(window)) {
        return ma;
    }
    
    for (size_t i = window - 1; i < data.size(); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window; ++j) {
            sum += data[i - j];
        }
        ma.push_back(sum / window);
    }
    
    return ma;
}

float ChartGenerator::calculateMean(const std::vector<float>& data) const {
    if (data.empty()) return 0.0f;
    
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    return sum / data.size();
}

float ChartGenerator::calculateStdDev(const std::vector<float>& data) const {
    if (data.size() < 2) return 0.0f;
    
    float mean = calculateMean(data);
    float sum_sq_diff = 0.0f;
    
    for (float value : data) {
        float diff = value - mean;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / (data.size() - 1));
}

void ChartGenerator::optimizeDataForVisualization(std::vector<float>& data, int max_points) const {
    if (data.size() <= static_cast<size_t>(max_points)) {
        return; // No optimization needed
    }
    
    std::vector<float> optimized_data;
    size_t step = data.size() / max_points;
    
    for (size_t i = 0; i < data.size(); i += step) {
        optimized_data.push_back(data[i]);
    }
    
    // Always include the last point
    if (!data.empty() && optimized_data.back() != data.back()) {
        optimized_data.push_back(data.back());
    }
    
    data = optimized_data;
}

// === Configuration Methods ===

void ChartGenerator::setTimeFrame(TimeFrame frame, int custom_points) {
    current_timeframe_ = frame;
    custom_point_limit_ = custom_points;
}

void ChartGenerator::setColorScheme(const std::string& scheme) {
    // This could be expanded to support different color schemes
    // For now, we'll just log the setting
    std::cout << "[ChartGenerator] Color scheme set to: " << scheme << std::endl;
}

void ChartGenerator::enableInteractivity(bool enable) {
    // This could be expanded to add/remove interactive features
    std::cout << "[ChartGenerator] Interactivity " << (enable ? "enabled" : "disabled") << std::endl;
}

std::string ChartGenerator::generateUniqueChartId() {
    return "chart_" + std::to_string(++chart_counter_);
}

// === Logging Methods ===

void ChartGenerator::logError(const std::string& message) {
    std::cerr << "[ChartGenerator ERROR] " << message << std::endl;
    last_error_ = message;
}

void ChartGenerator::logWarning(const std::string& message) {
    std::cout << "[ChartGenerator WARNING] " << message << std::endl;
}

void ChartGenerator::logInfo(const std::string& message) {
    std::cout << "[ChartGenerator INFO] " << message << std::endl;
}
