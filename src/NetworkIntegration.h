#ifndef NETWORK_INTEGRATION_H
#define NETWORK_INTEGRATION_H

#include <NeuroGen/Network.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/GridBlockUtils.cuh>
#include <NeuroGen/EnhancedLearningSystem.h>
#include <NeuroGen/LearningRuleConstants.h>

#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>

/**
 * Enhanced Network Manager integrating advanced learning mechanisms
 * This class extends the base Network functionality with sophisticated
 * neurobiological learning rules and homeostatic mechanisms
 */
class EnhancedNetworkManager {
private:
    // Core network and learning components
    std::unique_ptr<Network> base_network_;
    std::unique_ptr<EnhancedLearningSystem> learning_system_;
    
    // Network state tracking
    bool learning_enabled_;
    bool homeostasis_enabled_;
    bool reward_learning_enabled_;
    
    // Performance monitoring
    struct PerformanceMetrics {
        double total_simulation_time;
        double learning_update_time;
        double network_update_time;
        int total_timesteps;
        int learning_updates;
        float average_firing_rate;
        float network_stability_index;
        float learning_efficiency;
    };
    
    PerformanceMetrics metrics_;
    
    // Reward learning state
    float accumulated_reward_;
    float reward_decay_factor_;
    std::vector<float> reward_history_;
    
    // Network monitoring
    std::vector<float> activity_history_;
    std::vector<float> weight_change_history_;
    
public:
    /**
     * Constructor initializes enhanced network with learning capabilities
     */
    explicit EnhancedNetworkManager(const NetworkConfig& config) 
        : learning_enabled_(true)
        , homeostasis_enabled_(true) 
        , reward_learning_enabled_(true)
        , accumulated_reward_(0.0f)
        , reward_decay_factor_(0.99f) {
        
        // Initialize base network
        base_network_ = std::make_unique<Network>(config);
        
        // Initialize enhanced learning system
        int num_synapses = base_network_->getNumSynapses();
        int num_neurons = base_network_->getNumNeurons();
        
        learning_system_ = std::make_unique<EnhancedLearningSystem>(
            num_synapses, num_neurons);
        
        // Initialize metrics
        memset(&metrics_, 0, sizeof(PerformanceMetrics));
        
        std::cout << "Enhanced Network Manager initialized with " 
                  << num_neurons << " neurons and " 
                  << num_synapses << " synapses" << std::endl;
    }
    
    /**
     * Main simulation step with integrated learning
     */
    void simulateStep(float dt, float external_reward = 0.0f) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // ========================================
        // PHASE 1: BASE NETWORK SIMULATION
        // ========================================
        auto network_start = std::chrono::high_resolution_clock::now();
        
        // Update base network dynamics (spike generation, synaptic transmission, etc.)
        base_network_->update(dt);
        
        auto network_end = std::chrono::high_resolution_clock::now();
        auto network_duration = std::chrono::duration<double, std::milli>(
            network_end - network_start).count();
        
        // ========================================
        // PHASE 2: ENHANCED LEARNING UPDATES
        // ========================================
        if (learning_enabled_) {
            auto learning_start = std::chrono::high_resolution_clock::now();
            
            // Process reward signal
            float processed_reward = processRewardSignal(external_reward);
            
            // Get current network state
            GPUSynapse* synapses = base_network_->getGPUSynapses();
            GPUNeuronState* neurons = base_network_->getGPUNeurons();
            float current_time = base_network_->getCurrentTime();
            
            // Update learning system
            learning_system_->updateLearning(synapses, neurons, current_time, dt, processed_reward);
            
            auto learning_end = std::chrono::high_resolution_clock::now();
            auto learning_duration = std::chrono::duration<double, std::milli>(
                learning_end - learning_start).count();
            
            // Update metrics
            metrics_.learning_update_time += learning_duration;
            metrics_.learning_updates++;
        }
        
        // ========================================
        // PHASE 3: MONITORING AND ANALYSIS
        // ========================================
        updateNetworkMonitoring();
        
        // Update overall metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        metrics_.total_simulation_time += total_duration;
        metrics_.network_update_time += network_duration;
        metrics_.total_timesteps++;
    }
    
    /**
     * Set input values to the network
     */
    void setInputs(const std::vector<double>& inputs) {
        base_network_->setInputValues(inputs);
    }
    
    /**
     * Get output values from the network
     */
    std::vector<double> getOutputs() const {
        return base_network_->getOutputValues();
    }
    
    /**
     * Provide reward signal for learning
     */
    void provideReward(float reward) {
        if (reward_learning_enabled_) {
            learning_system_->setRewardSignal(reward);
            accumulated_reward_ += reward;
            reward_history_.push_back(reward);
            
            // Trigger protein synthesis for significant rewards
            if (std::abs(reward) > 0.5f) {
                learning_system_->triggerProteinSynthesis(std::abs(reward));
            }
        }
    }
    
    /**
     * Reset network for episodic learning
     */
    void resetEpisode() {
        if (learning_enabled_) {
            learning_system_->resetEpisode(true, true);
            accumulated_reward_ = 0.0f;
        }
        
        // Reset base network state if needed
        base_network_->resetDynamics();
    }
    
    /**
     * Enable/disable learning mechanisms
     */
    void setLearningEnabled(bool enabled) {
        learning_enabled_ = enabled;
        std::cout << "Learning " << (enabled ? "enabled" : "disabled") << std::endl;
    }
    
    void setHomeostasisEnabled(bool enabled) {
        homeostasis_enabled_ = enabled;
        std::cout << "Homeostasis " << (enabled ? "enabled" : "disabled") << std::endl;
    }
    
    void setRewardLearningEnabled(bool enabled) {
        reward_learning_enabled_ = enabled;
        std::cout << "Reward learning " << (enabled ? "enabled" : "disabled") << std::endl;
    }
    
    /**
     * Get comprehensive network statistics
     */
    struct NetworkStatistics {
        PerformanceMetrics performance;
        EnhancedLearningSystem::LearningStats learning;
        float average_reward;
        float network_efficiency;
        float learning_progress;
        int total_spikes;
        float weight_stability;
    };
    
    NetworkStatistics getStatistics() const {
        NetworkStatistics stats;
        
        // Performance metrics
        stats.performance = metrics_;
        
        // Learning statistics
        stats.learning = learning_system_->getStatistics();
        
        // Reward statistics
        if (!reward_history_.empty()) {
            float sum = 0.0f;
            for (float r : reward_history_) sum += r;
            stats.average_reward = sum / reward_history_.size();
        } else {
            stats.average_reward = 0.0f;
        }
        
        // Network efficiency metrics
        stats.network_efficiency = calculateNetworkEfficiency();
        stats.learning_progress = calculateLearningProgress();
        stats.total_spikes = base_network_->getTotalSpikes();
        stats.weight_stability = calculateWeightStability();
        
        return stats;
    }
    
    /**
     * Save network state and learning progress
     */
    bool saveNetworkState(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for saving: " << filename << std::endl;
            return false;
        }
        
        try {
            // Save network structure
            base_network_->saveToFile(filename + "_network.dat");
            
            // Save learning statistics
            auto stats = getStatistics();
            file.write(reinterpret_cast<const char*>(&stats), sizeof(NetworkStatistics));
            
            // Save reward history
            size_t history_size = reward_history_.size();
            file.write(reinterpret_cast<const char*>(&history_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(reward_history_.data()), 
                      history_size * sizeof(float));
            
            std::cout << "Network state saved to " << filename << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving network state: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * Load network state and learning progress
     */
    bool loadNetworkState(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for loading: " << filename << std::endl;
            return false;
        }
        
        try {
            // Load network structure
            if (!base_network_->loadFromFile(filename + "_network.dat")) {
                std::cerr << "Failed to load base network" << std::endl;
                return false;
            }
            
            // Load learning statistics
            NetworkStatistics stats;
            file.read(reinterpret_cast<char*>(&stats), sizeof(NetworkStatistics));
            metrics_ = stats.performance;
            
            // Load reward history
            size_t history_size;
            file.read(reinterpret_cast<char*>(&history_size), sizeof(size_t));
            reward_history_.resize(history_size);
            file.read(reinterpret_cast<char*>(reward_history_.data()), 
                     history_size * sizeof(float));
            
            std::cout << "Network state loaded from " << filename << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading network state: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * Print detailed performance report
     */
    void printPerformanceReport() const {
        auto stats = getStatistics();
        
        std::cout << "\n=== Enhanced Network Performance Report ===" << std::endl;
        std::cout << "Simulation Statistics:" << std::endl;
        std::cout << "  Total timesteps: " << stats.performance.total_timesteps << std::endl;
        std::cout << "  Total simulation time: " << stats.performance.total_simulation_time << " ms" << std::endl;
        std::cout << "  Average time per step: " 
                  << (stats.performance.total_simulation_time / stats.performance.total_timesteps) 
                  << " ms" << std::endl;
        
        std::cout << "\nLearning Statistics:" << std::endl;
        std::cout << "  Learning updates: " << stats.learning.plasticity_updates << std::endl;
        std::cout << "  Total weight change: " << stats.learning.total_weight_change << std::endl;
        std::cout << "  Average trace activity: " << stats.learning.average_trace_activity << std::endl;
        std::cout << "  Current dopamine level: " << stats.learning.current_dopamine_level << std::endl;
        std::cout << "  Prediction error: " << stats.learning.prediction_error << std::endl;
        
        std::cout << "\nNetwork Health:" << std::endl;
        std::cout << "  Network activity: " << stats.learning.network_activity << std::endl;
        std::cout << "  Average reward: " << stats.average_reward << std::endl;
        std::cout << "  Network efficiency: " << stats.network_efficiency << std::endl;
        std::cout << "  Learning progress: " << stats.learning_progress << std::endl;
        std::cout << "  Weight stability: " << stats.weight_stability << std::endl;
        std::cout << "  Total spikes: " << stats.total_spikes << std::endl;
        
        std::cout << "\nPerformance Breakdown:" << std::endl;
        float learning_percentage = (stats.performance.learning_update_time / 
                                   stats.performance.total_simulation_time) * 100.0f;
        float network_percentage = (stats.performance.network_update_time / 
                                  stats.performance.total_simulation_time) * 100.0f;
        
        std::cout << "  Network simulation: " << network_percentage << "%" << std::endl;
        std::cout << "  Learning updates: " << learning_percentage << "%" << std::endl;
        std::cout << "  Other overhead: " << (100.0f - learning_percentage - network_percentage) << "%" << std::endl;
        std::cout << "===========================================" << std::endl;
    }

private:
    /**
     * Process and condition reward signals
     */
    float processRewardSignal(float raw_reward) {
        // Apply temporal discounting
        accumulated_reward_ *= reward_decay_factor_;
        
        // Add current reward
        accumulated_reward_ += raw_reward;
        
        // Apply reward prediction error enhancement
        float processed_reward = raw_reward;
        
        // Normalize based on recent reward history
        if (reward_history_.size() > 10) {
            float recent_avg = 0.0f;
            int recent_count = std::min(10, (int)reward_history_.size());
            for (int i = reward_history_.size() - recent_count; i < reward_history_.size(); i++) {
                recent_avg += reward_history_[i];
            }
            recent_avg /= recent_count;
            
            // Enhance unexpected rewards
            processed_reward = raw_reward - recent_avg * 0.5f;
        }
        
        return processed_reward;
    }
    
    /**
     * Update network monitoring and diagnostics
     */
    void updateNetworkMonitoring() {
        // Track network activity
        float current_activity = base_network_->getAverageActivity();
        activity_history_.push_back(current_activity);
        
        // Limit history size
        if (activity_history_.size() > 1000) {
            activity_history_.erase(activity_history_.begin());
        }
        
        // Track learning progress
        if (learning_enabled_) {
            auto learning_stats = learning_system_->getStatistics();
            weight_change_history_.push_back(learning_stats.total_weight_change);
            
            if (weight_change_history_.size() > 1000) {
                weight_change_history_.erase(weight_change_history_.begin());
            }
        }
        
        // Update derived metrics
        if (metrics_.total_timesteps > 0) {
            metrics_.average_firing_rate = current_activity;
            metrics_.network_stability_index = calculateStabilityIndex();
            metrics_.learning_efficiency = calculateLearningEfficiency();
        }
    }
    
    /**
     * Calculate network efficiency metric
     */
    float calculateNetworkEfficiency() const {
        if (activity_history_.empty()) return 0.0f;
        
        // Efficiency based on activity variance and reward correlation
        float activity_variance = 0.0f;
        float activity_mean = 0.0f;
        
        for (float activity : activity_history_) {
            activity_mean += activity;
        }
        activity_mean /= activity_history_.size();
        
        for (float activity : activity_history_) {
            float diff = activity - activity_mean;
            activity_variance += diff * diff;
        }
        activity_variance /= activity_history_.size();
        
        // Lower variance with moderate activity indicates efficiency
        float efficiency = activity_mean / (1.0f + activity_variance);
        return std::min(1.0f, efficiency);
    }
    
    /**
     * Calculate learning progress metric
     */
    float calculateLearningProgress() const {
        if (weight_change_history_.size() < 10) return 0.0f;
        
        // Learning progress based on adaptation rate
        float recent_changes = 0.0f;
        float early_changes = 0.0f;
        
        int half_point = weight_change_history_.size() / 2;
        
        for (int i = 0; i < half_point; i++) {
            early_changes += std::abs(weight_change_history_[i]);
        }
        
        for (int i = half_point; i < weight_change_history_.size(); i++) {
            recent_changes += std::abs(weight_change_history_[i]);
        }
        
        // Progress indicates stabilization after initial learning
        if (early_changes > 0.0f) {
            return 1.0f - (recent_changes / early_changes);
        }
        
        return 0.0f;
    }
    
    /**
     * Calculate weight stability metric
     */
    float calculateWeightStability() const {
        if (weight_change_history_.size() < 10) return 0.0f;
        
        // Stability based on recent weight change variance
        float mean = 0.0f;
        int recent_count = std::min(50, (int)weight_change_history_.size());
        
        for (int i = weight_change_history_.size() - recent_count; 
             i < weight_change_history_.size(); i++) {
            mean += weight_change_history_[i];
        }
        mean /= recent_count;
        
        float variance = 0.0f;
        for (int i = weight_change_history_.size() - recent_count; 
             i < weight_change_history_.size(); i++) {
            float diff = weight_change_history_[i] - mean;
            variance += diff * diff;
        }
        variance /= recent_count;
        
        // Lower variance indicates higher stability
        return 1.0f / (1.0f + variance);
    }
    
    /**
     * Calculate network stability index
     */
    float calculateStabilityIndex() const {
        if (activity_history_.size() < 10) return 0.0f;
        
        // Stability based on activity oscillations
        float oscillation_measure = 0.0f;
        for (size_t i = 1; i < activity_history_.size(); i++) {
            float change = std::abs(activity_history_[i] - activity_history_[i-1]);
            oscillation_measure += change;
        }
        
        oscillation_measure /= (activity_history_.size() - 1);
        
        // Convert to stability (inverse of oscillations)
        return 1.0f / (1.0f + oscillation_measure);
    }
    
    /**
     * Calculate learning efficiency metric
     */
    float calculateLearningEfficiency() const {
        if (!learning_enabled_ || reward_history_.empty() || weight_change_history_.empty()) {
            return 0.0f;
        }
        
        // Efficiency based on reward per unit weight change
        float total_reward = 0.0f;
        float total_weight_change = 0.0f;
        
        for (float reward : reward_history_) {
            total_reward += reward;
        }
        
        for (float change : weight_change_history_) {
            total_weight_change += std::abs(change);
        }
        
        if (total_weight_change > 0.0f) {
            return std::max(0.0f, total_reward / total_weight_change);
        }
        
        return 0.0f;
    }
};

#endif // NETWORK_INTEGRATION_H