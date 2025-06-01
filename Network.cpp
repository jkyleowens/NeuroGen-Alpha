/**
 * @file Network.cpp
 * @brief Definitions for Network class and related functions
 */

#include "Network.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

// Constructor
Network::Network(const NetworkConfig& config) 
    : config_(config), 
      current_time_(0.0),
      rng_(std::random_device{}()),
      last_pruning_time_(0.0),
      last_neurogenesis_time_(0.0),
      bin_size_(50.0) {
    initializeSpatialBins();
    std::cout << "Creating network with initialized state" << std::endl;
}

// Destructor
Network::~Network() {
    std::cout << "Destroying network" << std::endl;
}

// Add a neuron to the network
size_t Network::addNeuron(std::shared_ptr<Neuron> neuron, const Position3D& position) {
    if (!neuron) {
        std::cerr << "Error: Cannot add null neuron" << std::endl;
        return SIZE_MAX; // Error indicator
    }
    size_t new_id = getNextNeuronId();
    if (new_id >= neurons_.size()) {
        neurons_.resize(new_id + 1);
        neuron_positions_.resize(new_id + 1);
        // ... resize other per-neuron vectors like spike_history_, last_activity_check_
        spike_history_.resize(new_id + 1);
        last_activity_check_.resize(new_id + 1, 0.0);
        external_currents_.resize(new_id + 1, 0.0);
    }
    neurons_[new_id] = neuron;
    neuron_positions_[new_id] = position;
    active_neuron_ids_.insert(new_id);
    return new_id; // Return the neuron ID
}

// Create a synapse between two neurons
bool Network::createSynapse(size_t pre_neuron_id, size_t post_neuron_id,
                          const std::string& post_compartment, size_t receptor_index,
                          double weight, double delay) { // Matched signature from .h
    if (pre_neuron_id >= neurons_.size() || post_neuron_id >= neurons_.size() || !neurons_[pre_neuron_id] || !neurons_[post_neuron_id]) {
        std::cerr << "Error: Invalid neuron IDs for synapse creation" << std::endl;
        return false;
    }
    
    // Use the Synapse constructor defined in Network.h
    synapses_.push_back(std::make_unique<Synapse>(pre_neuron_id, post_neuron_id, post_compartment, receptor_index, weight, delay));
    size_t synapse_idx = synapses_.size() - 1;
    outgoing_synapses_[pre_neuron_id].push_back(synapse_idx);
    incoming_synapses_[post_neuron_id].push_back(synapse_idx);
    return true;
}

// Inject current into a specific neuron
void Network::injectCurrent(size_t neuron_id, double current) { // Matched signature from .h
    if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) {
        std::cerr << "Error: Invalid neuron ID for current injection" << std::endl;
        return;
    }
    
    external_currents_[neuron_id] += current; // Or directly call neuron's method if preferred
    // neurons_[neuron_id]->injectCurrent(current); // If Neuron class has this method
}

// Execute a single timestep of the network
void Network::step(double dt) { // Fixed method name and signature
    current_time_ += dt; // Use provided dt
    processDelayedSpikes();

    // Update all neurons
    for (size_t id : active_neuron_ids_) {
        if(neurons_[id]){
            neurons_[id]->update(dt); // Use provided dt parameter
            if(external_currents_[id] != 0.0){
                neurons_[id]->injectCurrent(external_currents_[id]);
                external_currents_[id] = 0.0; // Reset after applying
            }
        }
    }
    
    // Process synaptic transmission for spikes that occurred in this step
    // This part needs to be careful about when spikes are considered "fired"
    // and how they are added to spike_queue_ vs processed immediately.
    // The current spike_queue_ logic in the original header seems to handle delayed events.
    // For immediate effects or STDP, direct processing might be needed.

    // For STDP, updatePlasticity might be called here or after spike processing
    if (config_.enable_stdp) {
        updatePlasticity();
    }

    // Neuromodulation decay
    if (config_.enable_neuromodulation) {
        neuromodulators_.decay(config_.dt); // Use config_.dt
    }

    // Periodic structural plasticity checks
    if (config_.enable_neurogenesis && (current_time_ - last_neurogenesis_time_) >= config_.neurogenesis_rate) { // Note: neurogenesis_rate might be better as an interval
        performNeurogenesis();
        last_neurogenesis_time_ = current_time_;
    }
    if (config_.enable_pruning && (current_time_ - last_pruning_time_) >= config_.pruning_check_interval) {
        performPruning();
        last_pruning_time_ = current_time_;
    }
    
    // Update layer-specific dynamics if applicable
    // updateLayeredNeurogenesisAndPruning(); // If this is a separate periodic call
}

// Run the network for a specified duration
void Network::run(double duration, double dt_sim) { // Renamed dt to dt_sim to avoid conflict with config_.dt
    double time = 0.0;
    while (time < duration) {
        step(dt_sim); // Call correct method with dt parameter
        time += dt_sim; // Increment by the simulation step, not necessarily config_.dt
        // Optional: Add logging or checks here
    }
}

// Calculate network statistics
Network::NetworkStats Network::calculateNetworkStats(double timeWindow) const {
    NetworkStats stats = {0.0, 0.0, 0.0, 0.0, 0, 0, 0.0};
    
    if (active_neuron_ids_.empty() || timeWindow <= 0.0) {
        stats.total_synapses = synapses_.size();
        stats.active_neurons = active_neuron_ids_.size();
        return stats;
    }

    // Calculate firing rates
    double cumulative_firing_rate = 0.0;
    std::vector<double> firing_rates;
    
    for (size_t id : active_neuron_ids_) {
        if (id < neurons_.size() && neurons_[id]) {
            double rate = neurons_[id]->getFiringRate(timeWindow);
            firing_rates.push_back(rate);
            cumulative_firing_rate += rate;
        }
    }

    stats.active_neurons = active_neuron_ids_.size();
    stats.mean_firing_rate = stats.active_neurons > 0 ? 
                            cumulative_firing_rate / stats.active_neurons : 0.0;

    // Count active synapses and calculate mean strength
    size_t active_synapses = 0;
    double total_strength = 0.0;
    for (const auto& syn_ptr : synapses_) {
        if (syn_ptr && syn_ptr->weight > config_.min_synaptic_weight) {
            active_synapses++;
            total_strength += syn_ptr->weight;
        }
    }
    
    stats.total_synapses = active_synapses;
    stats.mean_synaptic_strength = active_synapses > 0 ? 
                                  total_strength / active_synapses : 0.0;

    // Calculate connectivity
    stats.mean_connectivity = stats.active_neurons > 1 ? 
                             static_cast<double>(stats.total_synapses) / 
                             (stats.active_neurons * (stats.active_neurons - 1)) : 0.0;

    // Calculate network synchrony (correlation of firing rates)
    if (firing_rates.size() > 1) {
        double mean_rate = stats.mean_firing_rate;
        double variance = 0.0;
        for (double rate : firing_rates) {
            variance += (rate - mean_rate) * (rate - mean_rate);
        }
        variance /= firing_rates.size();
        
        // Synchrony is inverse of coefficient of variation
        stats.network_synchrony = variance > 0 ? 
                                 mean_rate / std::sqrt(variance) : 0.0;
        stats.network_synchrony = std::min(stats.network_synchrony, 1.0);
    }

    // Calculate E/I ratio
    double excitatory_rate = 0.0, inhibitory_rate = 0.0;
    size_t exc_count = 0, inh_count = 0;
    
    for (size_t id : active_neuron_ids_) {
        if (id < neurons_.size() && neurons_[id]) {
            // Determine neuron type based on layer assignment or other criteria
            // This is a simplified approach
            LayerType layer = getLayerType(id);
            if (layer == INPUT || layer == HIDDEN || layer == OUTPUT) {
                excitatory_rate += neurons_[id]->getFiringRate(timeWindow);
                exc_count++;
            } else {
                inhibitory_rate += neurons_[id]->getFiringRate(timeWindow);
                inh_count++;
            }
        }
    }
    
    if (exc_count > 0) excitatory_rate /= exc_count;
    if (inh_count > 0) inhibitory_rate /= inh_count;
    
    stats.excitation_inhibition_ratio = inhibitory_rate > 0 ? 
                                       excitatory_rate / inhibitory_rate : 
                                       excitatory_rate;

    return stats;
}

// === Layered Architecture Implementation ===

void Network::initializeLayeredArchitecture(size_t inputSize, size_t initialHiddenSize, size_t outputSize) {
    // Clear existing layer assignments
    inputLayerNeurons_.clear();
    hiddenLayerNeurons_.clear();
    outputLayerNeurons_.clear();
    
    // Create input layer neurons
    for (size_t i = 0; i < inputSize; ++i) {
        Position3D pos(i * 25.0, 0.0, 0.0); // Arrange in line
        std::string id = "input_" + std::to_string(i);
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::SENSORY, id, pos);
        size_t neuron_id = addNeuronToLayer(INPUT, neuron, pos);
        inputLayerNeurons_.push_back(neuron_id);
    }
    
    // Create hidden layer neurons
    for (size_t i = 0; i < initialHiddenSize; ++i) {
        Position3D pos(i * 15.0, 100.0, 0.0); // Arrange in middle layer
        std::string id = "hidden_" + std::to_string(i);
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
        size_t neuron_id = addNeuronToLayer(HIDDEN, neuron, pos);
        hiddenLayerNeurons_.push_back(neuron_id);
    }
    
    // Create output layer neurons
    for (size_t i = 0; i < outputSize; ++i) {
        Position3D pos(i * 25.0, 200.0, 0.0); // Arrange in output line
        std::string id = "output_" + std::to_string(i);
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::MOTOR, id, pos);
        size_t neuron_id = addNeuronToLayer(OUTPUT, neuron, pos);
        outputLayerNeurons_.push_back(neuron_id);
    }
    
    // Connect layers
    connectLayers();
}

void Network::evaluateInputLayerSize(size_t requiredInputSize) {
    if (requiredInputSize > inputLayerNeurons_.size()) {
        // Add more input neurons
        size_t neuronsToAdd = requiredInputSize - inputLayerNeurons_.size();
        for (size_t i = 0; i < neuronsToAdd; ++i) {
            Position3D pos((inputLayerNeurons_.size() + i) * 25.0, 0.0, 0.0);
            std::string id = "input_" + std::to_string(inputLayerNeurons_.size() + i);
            auto neuron = NeuronFactory::createNeuron(NeuronFactory::SENSORY, id, pos);
            size_t neuron_id = addNeuronToLayer(INPUT, neuron, pos);
            inputLayerNeurons_.push_back(neuron_id);
        }
    } else if (requiredInputSize < inputLayerNeurons_.size()) {
        // Remove excess input neurons
        size_t neuronsToRemove = inputLayerNeurons_.size() - requiredInputSize;
        for (size_t i = 0; i < neuronsToRemove; ++i) {
            size_t neuronToRemove = inputLayerNeurons_.back();
            removeNeuron(neuronToRemove);
            inputLayerNeurons_.pop_back();
        }
    }
}

void Network::evaluateOutputLayerSize(size_t requiredOutputSize) {
    if (requiredOutputSize > outputLayerNeurons_.size()) {
        // Add more output neurons
        size_t neuronsToAdd = requiredOutputSize - outputLayerNeurons_.size();
        for (size_t i = 0; i < neuronsToAdd; ++i) {
            Position3D pos((outputLayerNeurons_.size() + i) * 25.0, 200.0, 0.0);
            std::string id = "output_" + std::to_string(outputLayerNeurons_.size() + i);
            auto neuron = NeuronFactory::createNeuron(NeuronFactory::MOTOR, id, pos);
            size_t neuron_id = addNeuronToLayer(OUTPUT, neuron, pos);
            outputLayerNeurons_.push_back(neuron_id);
        }
    } else if (requiredOutputSize < outputLayerNeurons_.size()) {
        // Remove excess output neurons
        size_t neuronsToRemove = outputLayerNeurons_.size() - requiredOutputSize;
        for (size_t i = 0; i < neuronsToRemove; ++i) {
            size_t neuronToRemove = outputLayerNeurons_.back();
            removeNeuron(neuronToRemove);
            outputLayerNeurons_.pop_back();
        }
    }
}

void Network::optimizeHiddenLayer() {
    // Simple optimization: add neurons if network activity is high, remove if low
    auto stats = calculateNetworkStats(100.0);
    
    if (stats.mean_firing_rate > 50.0 && hiddenLayerNeurons_.size() < maxHiddenLayerSize_) {
        // High activity - add neuron
        Position3D pos(hiddenLayerNeurons_.size() * 15.0, 100.0, 0.0);
        std::string id = "hidden_" + std::to_string(hiddenLayerNeurons_.size());
        auto neuron = NeuronFactory::createNeuron(NeuronFactory::PYRAMIDAL_CORTICAL, id, pos);
        size_t neuron_id = addNeuronToLayer(HIDDEN, neuron, pos);
        hiddenLayerNeurons_.push_back(neuron_id);
        
        // Connect new neuron to nearby neurons in same layer
        createDistanceBasedConnections({neuron_id}, hiddenLayerNeurons_, 
                                     hiddenRecurrentProbability_, 
                                     hiddenRecurrentMaxDistance_, 
                                     hiddenRecurrentDecayConstant_);
    } else if (stats.mean_firing_rate < 5.0 && hiddenLayerNeurons_.size() > 1) {
        // Low activity - consider removing weakest neuron
        // Find neuron with lowest activity
        size_t weakestNeuron = hiddenLayerNeurons_[0];
        double lowestActivity = 0.0;
        
        for (size_t neuron_id : hiddenLayerNeurons_) {
            if (neuron_id < spike_history_.size()) {
                double activity = static_cast<double>(spike_history_[neuron_id].size());
                if (activity < lowestActivity || weakestNeuron == hiddenLayerNeurons_[0]) {
                    lowestActivity = activity;
                    weakestNeuron = neuron_id;
                }
            }
        }
        
        if (lowestActivity < 1.0) { // Very low activity
            auto it = std::find(hiddenLayerNeurons_.begin(), hiddenLayerNeurons_.end(), weakestNeuron);
            if (it != hiddenLayerNeurons_.end()) {
                hiddenLayerNeurons_.erase(it);
                removeNeuron(weakestNeuron);
            }
        }
    }
}

size_t Network::addNeuronToLayer(LayerType layer, std::shared_ptr<Neuron> neuron, const Position3D& position) {
    (void)layer; // Suppress unused parameter warning
    size_t neuron_id = addNeuron(neuron, position);
    if (neuron_id == SIZE_MAX) {
        return SIZE_MAX; // Error
    }
    
    return neuron_id;
}

void Network::connectLayers() {
    // Connect input to hidden layer
    createDistanceBasedConnections(inputLayerNeurons_, hiddenLayerNeurons_, 
                                 interLayerProbability_, 
                                 interLayerMaxDistance_, 
                                 interLayerDecayConstant_);
    
    // Connect hidden to output layer
    createDistanceBasedConnections(hiddenLayerNeurons_, outputLayerNeurons_, 
                                 interLayerProbability_, 
                                 interLayerMaxDistance_, 
                                 interLayerDecayConstant_);
    
    // Create recurrent connections within hidden layer
    createDistanceBasedConnections(hiddenLayerNeurons_, hiddenLayerNeurons_, 
                                 hiddenRecurrentProbability_, 
                                 hiddenRecurrentMaxDistance_, 
                                 hiddenRecurrentDecayConstant_);
    
    // Create recurrent connections within input layer
    createDistanceBasedConnections(inputLayerNeurons_, inputLayerNeurons_, 
                                 inputRecurrentProbability_, 
                                 inputRecurrentMaxDistance_, 
                                 inputRecurrentDecayConstant_);
    
    // Create recurrent connections within output layer
    createDistanceBasedConnections(outputLayerNeurons_, outputLayerNeurons_, 
                                 outputRecurrentProbability_, 
                                 outputRecurrentMaxDistance_, 
                                 outputRecurrentDecayConstant_);
}

void Network::createDistanceBasedConnections(const std::vector<size_t>& source, 
                                           const std::vector<size_t>& target, 
                                           double probability, 
                                           double maxDistance, 
                                           double decayConstant) {
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    
    for (size_t src_id : source) {
        for (size_t tgt_id : target) {
            if (src_id == tgt_id) continue; // No self-connections
            
            if (src_id >= neuron_positions_.size() || tgt_id >= neuron_positions_.size()) continue;
            
            double distance = neuron_positions_[src_id].distanceTo(neuron_positions_[tgt_id]);
            
            if (distance <= maxDistance) {
                double distance_prob = probability * std::exp(-distance / decayConstant);
                
                if (prob_dist(rng_) < distance_prob) {
                    double weight = 0.1 + 0.1 * prob_dist(rng_); // Random weight 0.1-0.2
                    double delay = 1.0 + (distance / 100.0); // Delay based on distance
                    createSynapse(src_id, tgt_id, "dendrite", 0, weight, delay);
                }
            }
        }
    }
}

void Network::updateLayeredNeurogenesisAndPruning() {
    // Update hidden layer size based on activity
    optimizeHiddenLayer();
}

void Network::setInputValues(const std::vector<double>& inputs) {
    evaluateInputLayerSize(inputs.size());
    
    for (size_t i = 0; i < inputs.size() && i < inputLayerNeurons_.size(); ++i) {
        injectCurrent(inputLayerNeurons_[i], inputs[i] * 10.0); // Scale input
    }
}

std::vector<double> Network::getOutputValues() const {
    std::vector<double> outputs;
    outputs.reserve(outputLayerNeurons_.size());
    
    for (size_t neuron_id : outputLayerNeurons_) {
        if (neuron_id < neurons_.size() && neurons_[neuron_id]) {
            // Use recent firing rate as output
            double firingRate = neurons_[neuron_id]->getFiringRate(50.0); // 50ms window
            outputs.push_back(firingRate);
        } else {
            outputs.push_back(0.0);
        }
    }
    
    return outputs;
}

Network::LayerType Network::getLayerType(size_t neuron_id) const {
    if (std::find(inputLayerNeurons_.begin(), inputLayerNeurons_.end(), neuron_id) != inputLayerNeurons_.end()) {
        return INPUT;
    }
    if (std::find(hiddenLayerNeurons_.begin(), hiddenLayerNeurons_.end(), neuron_id) != hiddenLayerNeurons_.end()) {
        return HIDDEN;
    }
    if (std::find(outputLayerNeurons_.begin(), outputLayerNeurons_.end(), neuron_id) != outputLayerNeurons_.end()) {
        return OUTPUT;
    }
    return HIDDEN; // Default to hidden for unassigned neurons
}

// === STDP Implementation ===

void Network::updatePlasticity() {
    if (!config_.enable_stdp) return;
    
    const double tau_plus = config_.stdp_tau_pre;
    const double tau_minus = config_.stdp_tau_post;
    const double A_plus = config_.stdp_learning_rate;
    const double A_minus = config_.stdp_learning_rate * 1.05;
    
    for (auto& synapse : synapses_) {
        if (!synapse) continue;
        
        size_t pre_id = synapse->pre_neuron_id;
        size_t post_id = synapse->post_neuron_id;
        
        if (pre_id >= neurons_.size() || post_id >= neurons_.size() || 
            !neurons_[pre_id] || !neurons_[post_id]) continue;
        
        // Apply STDP to all connections, not just intra-layer
        double pre_spike_time = neurons_[pre_id]->getLastSpikeTime();
        double post_spike_time = neurons_[post_id]->getLastSpikeTime();
        
        // Only apply if both neurons have spiked recently
        if (pre_spike_time > current_time_ - 100.0 && post_spike_time > current_time_ - 100.0) {
            double dt = post_spike_time - pre_spike_time;
            double weight_change = 0.0;
            
            if (std::abs(dt) < 50.0) { // 50ms window
                if (dt > 0 && dt < 20.0) {
                    // Potentiation
                    weight_change = A_plus * std::exp(-dt / tau_plus);
                } else if (dt < 0 && dt > -20.0) {
                    // Depression
                    weight_change = -A_minus * std::exp(dt / tau_minus);
                }
                
                // Apply neuromodulation
                if (config_.enable_neuromodulation) {
                    double modulation = getModulatedLearningRate(1.0);
                    weight_change *= modulation;
                }
                
                // Update weight with bounds
                synapse->weight += weight_change;
                synapse->weight = std::max(config_.min_synaptic_weight, 
                                         std::min(config_.max_synaptic_weight, synapse->weight));
                
                // Update activity metrics
                synapse->activity_metric = 0.9 * synapse->activity_metric + 
                                         0.1 * std::abs(weight_change);
                
                if (weight_change > 0) {
                    synapse->last_potentiation = current_time_;
                }
            }
        }
    }
}

// === Missing Method Implementations ===

void Network::performSynaptogenesis() {
    if (!config_.enable_neurogenesis) return;
    
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<> weight_dist(config_.min_synaptic_weight, 
                                                config_.max_synaptic_weight * 0.5);
    
    // Calculate activity levels for all neurons
    std::vector<double> activity_levels(neurons_.size(), 0.0);
    for (size_t i = 0; i < neurons_.size(); ++i) {
        if (neurons_[i]) {
            activity_levels[i] = neurons_[i]->getFiringRate(config_.spike_correlation_window);
        }
    }
    
    // Try to form new synapses between active neurons
    for (size_t pre_id : active_neuron_ids_) {
        if (pre_id >= neurons_.size() || !neurons_[pre_id]) continue;
        
        // Only consider forming synapses from moderately active neurons
        if (activity_levels[pre_id] < 1.0 || activity_levels[pre_id] > 50.0) continue;
        
        // Find nearby neurons within connection distance
        auto nearby_neurons = getNeuronsWithinDistance(
            neuron_positions_[pre_id], config_.max_connection_distance);
        
        for (size_t post_id : nearby_neurons) {
            if (pre_id == post_id || post_id >= neurons_.size() || !neurons_[post_id]) continue;
            
            // Check if connection already exists
            bool connection_exists = false;
            auto it = outgoing_synapses_.find(pre_id);
            if (it != outgoing_synapses_.end()) {
                for (size_t syn_idx : it->second) {
                    if (syn_idx < synapses_.size() && 
                        synapses_[syn_idx]->post_neuron_id == post_id) {
                        connection_exists = true;
                        break;
                    }
                }
            }
            
            if (!connection_exists) {
                // Calculate connection probability based on activity correlation
                double correlation = calculateSpikeCorrelation(pre_id, post_id, 
                                                             config_.spike_correlation_window);
                double distance = neuron_positions_[pre_id].distanceTo(neuron_positions_[post_id]);
                double distance_factor = std::exp(-distance / config_.distance_decay_constant);
                
                double connection_prob = config_.connection_probability_base * 
                                       distance_factor * (1.0 + correlation);
                
                if (prob_dist(rng_) < connection_prob) {
                    double weight = weight_dist(rng_);
                    double delay = 1.0 + (distance / 1000.0); // 1m/s conduction velocity
                    
                    if (createSynapse(pre_id, post_id, "dendrite", 0, weight, delay)) {
                        logStructuralChange("SYNAPSE_FORMED", 
                            "Connection " + std::to_string(pre_id) + "->" + 
                            std::to_string(post_id) + " weight=" + std::to_string(weight));
                    }
                }
            }
        }
    }
}


void Network::performNeurogenesis() {
    if (!config_.enable_neurogenesis || active_neuron_ids_.size() >= config_.max_neurons) {
        return;
    }
    
    // Calculate regional activity levels
    auto regional_activity = getRegionalActivity(5, 5, 100.0);
    
    // Find regions with activity imbalance
    double mean_activity = 0.0;
    for (double activity : regional_activity) {
        mean_activity += activity;
    }
    mean_activity /= regional_activity.size();
    
    std::uniform_real_distribution<> pos_dist_x(0.0, config_.network_width);
    std::uniform_real_distribution<> pos_dist_y(0.0, config_.network_height);
    std::uniform_real_distribution<> pos_dist_z(0.0, config_.network_depth);
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    
    // Create new neurons in underactive regions
    for (size_t i = 0; i < regional_activity.size(); ++i) {
        if (regional_activity[i] < config_.activity_threshold_low * mean_activity) {
            
            if (prob_dist(rng_) < config_.neurogenesis_rate) {
                // Generate position in the underactive region
                size_t x_bin = i % 5;
                size_t y_bin = i / 5;
                
                double x = (x_bin * config_.network_width / 5.0) + 
                          pos_dist_x(rng_) * (config_.network_width / 5.0);
                double y = (y_bin * config_.network_height / 5.0) + 
                          pos_dist_y(rng_) * (config_.network_height / 5.0);
                double z = pos_dist_z(rng_);
                
                Position3D new_pos(x, y, z);
                std::string neuron_id = "neurogenesis_" + std::to_string(current_time_) + 
                                      "_" + std::to_string(active_neuron_ids_.size());
                
                // Create new neuron (80% excitatory, 20% inhibitory)
                auto new_neuron = NeuronFactory::createNeuron(
                    prob_dist(rng_) < 0.8 ? NeuronFactory::PYRAMIDAL_CORTICAL : 
                                          NeuronFactory::INTERNEURON_BASKET,
                    neuron_id, new_pos);
                
                size_t new_id = addNeuron(new_neuron, new_pos);
                
                // Connect to nearby neurons
                auto nearby_neurons = getNeuronsWithinDistance(new_pos, 100.0);
                for (size_t nearby_id : nearby_neurons) {
                    if (nearby_id != new_id && prob_dist(rng_) < 0.1) {
                        double weight = 0.1 + 0.1 * prob_dist(rng_);
                        createSynapse(new_id, nearby_id, "dendrite", 0, weight);
                        createSynapse(nearby_id, new_id, "dendrite", 0, weight);
                    }
                }
                
                logStructuralChange("NEURON_BORN", 
                    "New neuron " + std::to_string(new_id) + " at (" + 
                    std::to_string(x) + "," + std::to_string(y) + ")");
                
                break; // Only create one neuron per step
            }
        }
    }
}

void Network::performPruning() {
    if (!config_.enable_pruning) return;
    
    std::vector<size_t> synapses_to_remove;
    std::vector<size_t> neurons_to_remove;
    
    // Prune weak synapses
    for (size_t i = 0; i < synapses_.size(); ++i) {
        if (!synapses_[i]) continue;
        
        auto& synapse = synapses_[i];
        
        // Check if synapse is weak and hasn't been potentiated recently
        double time_since_potentiation = current_time_ - synapse->last_potentiation;
        bool is_weak = synapse->weight < config_.synapse_pruning_threshold;
        bool is_inactive = synapse->activity_metric < 0.01;
        bool is_old_without_potentiation = time_since_potentiation > 1000.0; // 1 second
        
        if (is_weak && (is_inactive || is_old_without_potentiation)) {
            synapses_to_remove.push_back(i);
        }
    }
    
    // Remove weak synapses
    for (auto it = synapses_to_remove.rbegin(); it != synapses_to_remove.rend(); ++it) {
        size_t syn_idx = *it;
        if (syn_idx < synapses_.size() && synapses_[syn_idx]) {
            size_t pre_id = synapses_[syn_idx]->pre_neuron_id;
            size_t post_id = synapses_[syn_idx]->post_neuron_id;
            
            // Remove from outgoing/incoming lists
            auto out_it = outgoing_synapses_.find(pre_id);
            if (out_it != outgoing_synapses_.end()) {
                auto& vec = out_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), syn_idx), vec.end());
            }
            
            auto in_it = incoming_synapses_.find(post_id);
            if (in_it != incoming_synapses_.end()) {
                auto& vec = in_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), syn_idx), vec.end());
            }
            
            logStructuralChange("SYNAPSE_PRUNED", 
                "Removed synapse " + std::to_string(pre_id) + "->" + 
                std::to_string(post_id) + " weight=" + std::to_string(synapses_[syn_idx]->weight));
            
            synapses_[syn_idx].reset();
        }
    }
    
    // Prune inactive neurons (but keep minimum population)
    if (active_neuron_ids_.size() > 10) {
        for (size_t neuron_id : active_neuron_ids_) {
            if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) continue;
            
            double activity = neurons_[neuron_id]->getFiringRate(config_.synapse_activity_window);
            
            // Remove neurons with very low activity and few connections
            if (activity < config_.neuron_pruning_threshold) {
                size_t connection_count = 0;
                auto out_it = outgoing_synapses_.find(neuron_id);
                auto in_it = incoming_synapses_.find(neuron_id);
                
                if (out_it != outgoing_synapses_.end()) connection_count += out_it->second.size();
                if (in_it != incoming_synapses_.end()) connection_count += in_it->second.size();
                
                if (connection_count < 2) {
                    neurons_to_remove.push_back(neuron_id);
                }
            }
        }
        
        // Remove inactive neurons (limit to 1 per pruning cycle)
        if (!neurons_to_remove.empty()) {
            size_t neuron_to_remove = neurons_to_remove[0];
            removeNeuron(neuron_to_remove);
        }
    }
}

void Network::releaseNeuromodulator(const std::string& type, double amount) {
    if (type == "dopamine") {
        neuromodulators_.releaseDopamine(amount);
    } else if (type == "acetylcholine") {
        neuromodulators_.releaseAcetylcholine(amount);
    } else if (type == "norepinephrine") {
        neuromodulators_.releaseNorepinephrine(amount);
    } else if (type == "serotonin") {
        neuromodulators_.releaseSerotonin(amount);
    }
}

void Network::processDelayedSpikes() {
    // Process spikes in the delay queue
    while (!spike_queue_.empty() && spike_queue_.top().timestamp <= current_time_) {
        const SpikeEvent& spike = spike_queue_.top();
        
        // Apply synaptic inputs for this spike
        auto it = outgoing_synapses_.find(spike.neuron_id);
        if (it != outgoing_synapses_.end()) {
            for (size_t synapse_idx : it->second) {
                if (synapse_idx < synapses_.size() && synapses_[synapse_idx]) {
                    const auto& synapse = synapses_[synapse_idx];
                    if (synapse->post_neuron_id < neurons_.size() && neurons_[synapse->post_neuron_id]) {
                        neurons_[synapse->post_neuron_id]->addSynapticInput(
                            synapse->post_compartment, 
                            synapse->receptor_index, 
                            current_time_
                        );
                    }
                }
            }
        }
        
        spike_queue_.pop();
    }
}

size_t Network::getNextNeuronId() {
    if (!available_neuron_ids_.empty()) {
        size_t id = *available_neuron_ids_.begin();
        available_neuron_ids_.erase(available_neuron_ids_.begin());
        return id;
    }
    return neurons_.size();
}

double Network::getModulatedLearningRate(double base_rate) const {
    double modulation = 1.0;
    modulation += config_.modulation_strength * neuromodulators_.dopamine;
    modulation += 0.5 * config_.modulation_strength * neuromodulators_.acetylcholine;
    return base_rate * std::max(0.1, modulation);
}

void Network::logStructuralChange(const std::string& type, const std::string& details) const {
    // Simple logging implementation
    std::cout << "[" << current_time_ << "ms] " << type << ": " << details << std::endl;
}

void Network::cleanupAfterNeuronRemoval(size_t neuron_id) {
    available_neuron_ids_.insert(neuron_id);
    active_neuron_ids_.erase(neuron_id);
    
    // Remove from layer vectors
    auto removeFromVector = [neuron_id](std::vector<size_t>& vec) {
        auto it = std::find(vec.begin(), vec.end(), neuron_id);
        if (it != vec.end()) {
            vec.erase(it);
        }
    };
    
    removeFromVector(inputLayerNeurons_);
    removeFromVector(hiddenLayerNeurons_);
    removeFromVector(outputLayerNeurons_);
}

void Network::initializeSpatialBins() {
    // Initialize spatial binning for efficient neighbor searching
    int x_bins = static_cast<int>(config_.network_width / bin_size_) + 1;
    int y_bins = static_cast<int>(config_.network_height / bin_size_) + 1;
    int z_bins = static_cast<int>(config_.network_depth / bin_size_) + 1;
    
    spatial_bins_.resize(x_bins);
    for (auto& x_layer : spatial_bins_) {
        x_layer.resize(y_bins);
        for (auto& y_layer : x_layer) {
            y_layer.resize(z_bins);
        }
    }
}

// Missing method implementations

bool Network::removeNeuron(size_t neuron_id) {
    if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) {
        std::cerr << "Error: Invalid neuron ID for removal: " << neuron_id << std::endl;
        return false;
    }
    
    // Remove all synapses connected to this neuron
    auto it = synapses_.begin();
    while (it != synapses_.end()) {
        if ((*it)->pre_neuron_id == neuron_id || (*it)->post_neuron_id == neuron_id) {
            it = synapses_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Clear the synapse lists for this neuron
    outgoing_synapses_[neuron_id].clear();
    incoming_synapses_[neuron_id].clear();
    
    // Remove neuron
    neurons_[neuron_id].reset();
    
    // Clean up data structures
    cleanupAfterNeuronRemoval(neuron_id);
    
    logStructuralChange("NEURON_REMOVED", "Neuron " + std::to_string(neuron_id) + " removed");
    return true;
}

std::vector<Synapse*> Network::getOutgoingSynapses(size_t neuron_id) {
    std::vector<Synapse*> outgoing;
    
    if (neuron_id >= neurons_.size()) {
        std::cerr << "Error: Invalid neuron ID: " << neuron_id << std::endl;
        return outgoing;
    }
    
    auto it = outgoing_synapses_.find(neuron_id);
    if (it != outgoing_synapses_.end()) {
        for (size_t synapse_idx : it->second) {
            if (synapse_idx < synapses_.size()) {
                outgoing.push_back(synapses_[synapse_idx].get());
            }
        }
    }
    return outgoing;
}

std::vector<double> Network::getRegionalActivity(size_t x_bins, size_t y_bins, double time_window) const {
    std::vector<double> regional_activity;
    
    // Calculate bins
    double x_step = config_.network_width / x_bins;
    double y_step = config_.network_height / y_bins;
    
    regional_activity.resize(x_bins * y_bins, 0.0);
    
    // For each bin, calculate average activity of neurons in that region
    for (size_t x = 0; x < x_bins; ++x) {
        for (size_t y = 0; y < y_bins; ++y) {
            double min_x = x * x_step;
            double max_x = (x + 1) * x_step;
            double min_y = y * y_step;
            double max_y = (y + 1) * y_step;
            
            double total_activity = 0.0;
            size_t neuron_count = 0;
            
            // Check all neurons
            for (size_t id = 0; id < neurons_.size(); ++id) {
                if (neurons_[id] && id < neuron_positions_.size()) {
                    const Position3D& pos = neuron_positions_[id];
                    if (pos.x >= min_x && pos.x < max_x && pos.y >= min_y && pos.y < max_y) {
                        total_activity += neurons_[id]->getFiringRate(time_window);
                        neuron_count++;
                    }
                }
            }
            
            regional_activity[x * y_bins + y] = neuron_count > 0 ? total_activity / neuron_count : 0.0;
        }
    }
    
    return regional_activity;
}

// NetworkBuilder implementation
NetworkBuilder::NetworkBuilder() : connectionProbability_(0.0) {
    // Initialize with default config
    config_ = NetworkConfig();
}

NetworkBuilder& NetworkBuilder::setConfig(const NetworkConfig& config) {
    config_ = config;
    return *this;
}

NetworkBuilder& NetworkBuilder::addNeuronPopulation(NeuronFactory::NeuronType type,
                                                   unsigned long count,
                                                   const Position3D& position,
                                                   double radius) {
    NeuronPopulation pop;
    pop.type = type;
    pop.count = count;
    pop.position = position;
    pop.radius = radius;
    neuronPopulations_.push_back(pop);
    return *this;
}

NetworkBuilder& NetworkBuilder::addRandomConnections(double probability) {
    connectionProbability_ = std::max(0.0, std::min(1.0, probability)); // Clamp to [0,1]
    return *this;
}

std::shared_ptr<Network> NetworkBuilder::build() {
    auto network = std::make_shared<Network>(config_);
    
    // Create all neuron populations
    std::vector<std::vector<size_t>> population_ids; // Track neurons by population
    
    for (const auto& pop : neuronPopulations_) {
        std::vector<size_t> pop_neurons;
        
        for (unsigned long i = 0; i < pop.count; ++i) {
            Position3D neuron_pos = randomizePosition(pop.position, pop.radius);
            std::string neuron_id = "neuron_" + std::to_string(i) + "_pop_" + std::to_string(population_ids.size());
            
            auto neuron = NeuronFactory::createNeuron(pop.type, neuron_id, neuron_pos);
            size_t id = network->addNeuron(neuron, neuron_pos);
            pop_neurons.push_back(id);
        }
        
        population_ids.push_back(pop_neurons);
    }
    
    // Create random connections if specified
    if (connectionProbability_ > 0.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Get all neuron IDs
        std::vector<size_t> all_neurons;
        for (const auto& pop : population_ids) {
            all_neurons.insert(all_neurons.end(), pop.begin(), pop.end());
        }
        
        // Create connections
        for (size_t i = 0; i < all_neurons.size(); ++i) {
            for (size_t j = 0; j < all_neurons.size(); ++j) {
                if (i != j && dis(gen) < connectionProbability_) {
                    // Create synapse with default weight
                    double weight = 0.5; // Default synaptic weight
                    network->createSynapse(all_neurons[i], all_neurons[j], "dendrite", 0, weight);
                }
            }
        }
    }
    
    return network;
}

Position3D NetworkBuilder::randomizePosition(const Position3D& center, double radius) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Generate random point in sphere using rejection sampling
    Position3D pos;
    do {
        pos.x = center.x + radius * dis(gen);
        pos.y = center.y + radius * dis(gen);
        pos.z = center.z + radius * dis(gen);
    } while (pos.distanceTo(center) > radius);
    
    return pos;
}

// Add these missing method implementations to the end of Network.cpp
// before the NetworkBuilder implementation section

// === Missing Method Implementations ===

std::vector<size_t> Network::getNeuronsWithinDistance(const Position3D& center, double radius) const {
    std::vector<size_t> nearby_neurons;
    
    for (size_t id : active_neuron_ids_) {
        if (id < neuron_positions_.size()) {
            double distance = neuron_positions_[id].distanceTo(center);
            if (distance <= radius) {
                nearby_neurons.push_back(id);
            }
        }
    }
    
    return nearby_neurons;
}

double Network::calculateSpikeCorrelation(size_t neuron1, size_t neuron2, double time_window) const {
    if (neuron1 >= spike_history_.size() || neuron2 >= spike_history_.size()) {
        return 0.0;
    }
    
    const auto& spikes1 = spike_history_[neuron1];
    const auto& spikes2 = spike_history_[neuron2];
    
    if (spikes1.empty() || spikes2.empty()) {
        return 0.0;
    }
    
    double correlation = 0.0;
    double window_start = current_time_ - time_window;
    
    // Count spikes in the time window for each neuron
    int count1 = 0, count2 = 0;
    for (double spike_time : spikes1) {
        if (spike_time >= window_start) count1++;
    }
    for (double spike_time : spikes2) {
        if (spike_time >= window_start) count2++;
    }
    
    // Simple correlation based on spike count similarity
    // Higher correlation when both neurons have similar spike counts
    if (count1 + count2 > 0) {
        correlation = 1.0 - std::abs(count1 - count2) / static_cast<double>(count1 + count2);
    }
    
    // Also check for temporal correlation by looking for coincident spikes
    double coincidence_bonus = 0.0;
    int coincident_spikes = 0;
    const double coincidence_window = 5.0; // ms
    
    for (double spike1 : spikes1) {
        if (spike1 >= window_start) {
            for (double spike2 : spikes2) {
                if (spike2 >= window_start && std::abs(spike1 - spike2) <= coincidence_window) {
                    coincident_spikes++;
                    break; // Only count each spike once
                }
            }
        }
    }
    
    if (std::max(count1, count2) > 0) {
        coincidence_bonus = static_cast<double>(coincident_spikes) / std::max(count1, count2);
    }
    
    // Combine count-based and coincidence-based correlation
    correlation = 0.7 * correlation + 0.3 * coincidence_bonus;
    
    return std::max(0.0, std::min(correlation, 1.0)); // Clamp to [0,1]
}

// === Additional missing stub implementations ===

std::shared_ptr<Neuron> Network::getNeuron(size_t neuron_id) const {
    if (neuron_id < neurons_.size()) {
        return neurons_[neuron_id];
    }
    return nullptr;
}

Position3D Network::getNeuronPosition(size_t neuron_id) const {
    if (neuron_id < neuron_positions_.size()) {
        return neuron_positions_[neuron_id];
    }
    return Position3D(0.0, 0.0, 0.0);
}

void Network::setNeuronPosition(size_t neuron_id, const Position3D& position) {
    if (neuron_id < neuron_positions_.size()) {
        neuron_positions_[neuron_id] = position;
        // Note: You might want to update spatial bins here if using spatial indexing
    }
}

bool Network::removeSynapse(size_t pre_neuron_id, size_t post_neuron_id,
                           const std::string& post_compartment, size_t receptor_index) {
    // Find and remove the specific synapse
    for (auto it = synapses_.begin(); it != synapses_.end(); ++it) {
        if ((*it) && (*it)->pre_neuron_id == pre_neuron_id && 
            (*it)->post_neuron_id == post_neuron_id &&
            (*it)->post_compartment == post_compartment &&
            (*it)->receptor_index == receptor_index) {
            
            // Remove from outgoing/incoming lists
            auto out_it = outgoing_synapses_.find(pre_neuron_id);
            if (out_it != outgoing_synapses_.end()) {
                size_t synapse_idx = it - synapses_.begin();
                auto& vec = out_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), synapse_idx), vec.end());
            }
            
            auto in_it = incoming_synapses_.find(post_neuron_id);
            if (in_it != incoming_synapses_.end()) {
                size_t synapse_idx = it - synapses_.begin();
                auto& vec = in_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), synapse_idx), vec.end());
            }
            
            synapses_.erase(it);
            return true;
        }
    }
    return false;
}

std::vector<Synapse*> Network::getIncomingSynapses(size_t neuron_id) {
    std::vector<Synapse*> incoming;
    
    if (neuron_id >= neurons_.size()) {
        return incoming;
    }
    
    auto it = incoming_synapses_.find(neuron_id);
    if (it != incoming_synapses_.end()) {
        for (size_t synapse_idx : it->second) {
            if (synapse_idx < synapses_.size() && synapses_[synapse_idx]) {
                incoming.push_back(synapses_[synapse_idx].get());
            }
        }
    }
    return incoming;
}

void Network::reset() {
    current_time_ = 0.0;
    last_pruning_time_ = 0.0;
    last_neurogenesis_time_ = 0.0;
    
    // Reset all neurons
    for (auto& neuron : neurons_) {
        if (neuron) {
            neuron->reset();
        }
    }
    
    // Clear spike queues and history
    while (!spike_queue_.empty()) {
        spike_queue_.pop();
    }
    
    for (auto& history : spike_history_) {
        history.clear();
    }
    
    // Reset neuromodulators
    neuromodulators_ = NeuromodulatorSystem();
    
    // Clear external currents
    std::fill(external_currents_.begin(), external_currents_.end(), 0.0);
}

void Network::stimulateNeuron(size_t neuron_id, const std::vector<double>& spike_times) {
    if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) {
        return;
    }
    
    // Add spike events to the queue
    for (double spike_time : spike_times) {
        spike_queue_.push(SpikeEvent(neuron_id, spike_time));
    }
}

void Network::addExternalInput(size_t neuron_id, const std::string& compartment,
                              size_t receptor_idx, double spike_time) {
    if (neuron_id < neurons_.size() && neurons_[neuron_id]) {
        neurons_[neuron_id]->addSynapticInput(compartment, receptor_idx, spike_time);
    }
}

bool Network::exportToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple export format - this could be expanded
    file << "# Network Export\n";
    file << "neurons=" << neurons_.size() << "\n";
    file << "synapses=" << synapses_.size() << "\n";
    file << "time=" << current_time_ << "\n";
    
    file.close();
    return true;
}

bool Network::importFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple import - this would need to be expanded for full functionality
    std::string line;
    while (std::getline(file, line)) {
        // Parse network data
        // This is just a placeholder
    }
    
    file.close();
    return true;
}

bool Network::loadConfig(const std::string& filename) {
    // Placeholder - would load JSON config if enabled
    (void)filename; // Suppress unused parameter warning
    return false;
}

bool Network::saveConfig(const std::string& filename) const {
    // Placeholder - would save JSON config if enabled
    (void)filename; // Suppress unused parameter warning
    return false;
}

void Network::updateSpatialBins() {
    // Clear all bins
    for (auto& x_layer : spatial_bins_) {
        for (auto& y_layer : x_layer) {
            for (auto& bin : y_layer) {
                bin.neuron_ids.clear();
            }
        }
    }
    
    // Reassign neurons to bins based on current positions
    for (size_t id : active_neuron_ids_) {
        if (id < neuron_positions_.size()) {
            const Position3D& pos = neuron_positions_[id];
            
            int x_bin = static_cast<int>(pos.x / bin_size_);
            int y_bin = static_cast<int>(pos.y / bin_size_);
            int z_bin = static_cast<int>(pos.z / bin_size_);
            
            // Clamp to valid ranges
            x_bin = std::max(0, std::min(x_bin, static_cast<int>(spatial_bins_.size()) - 1));
            y_bin = std::max(0, std::min(y_bin, static_cast<int>(spatial_bins_[0].size()) - 1));
            z_bin = std::max(0, std::min(z_bin, static_cast<int>(spatial_bins_[0][0].size()) - 1));
            
            spatial_bins_[x_bin][y_bin][z_bin].neuron_ids.push_back(id);
        }
    }
}

double Network::calculateConnectionProbability(size_t pre_id, size_t post_id) const {
    if (pre_id >= neuron_positions_.size() || post_id >= neuron_positions_.size()) {
        return 0.0;
    }
    
    double distance = neuron_positions_[pre_id].distanceTo(neuron_positions_[post_id]);
    if (distance > config_.max_connection_distance) {
        return 0.0;
    }
    
    // Distance-based probability
    double distance_factor = std::exp(-distance / config_.distance_decay_constant);
    
    // Activity-based modulation
    double activity_factor = 1.0;
    if (pre_id < neurons_.size() && post_id < neurons_.size() && 
        neurons_[pre_id] && neurons_[post_id]) {
        double pre_activity = neurons_[pre_id]->getFiringRate(config_.spike_correlation_window);
        double post_activity = neurons_[post_id]->getFiringRate(config_.spike_correlation_window);
        activity_factor = 1.0 + 0.1 * (pre_activity + post_activity) / 2.0;
    }
    
    return config_.connection_probability_base * distance_factor * activity_factor;
}

void Network::updateEligibilityTraces(double dt) {
    const double decay_factor = std::exp(-dt / config_.eligibility_decay);
    
    for (auto& synapse : synapses_) {
        if (synapse) {
            synapse->eligibility_trace *= decay_factor;
        }
    }
}

bool Network::shouldPruneNeuron(size_t neuron_id) const {
    if (neuron_id >= neurons_.size() || !neurons_[neuron_id]) {
        return true; // Remove invalid neurons
    }
    
    // Check activity level
    double activity = neurons_[neuron_id]->getFiringRate(config_.synapse_activity_window);
    if (activity < config_.neuron_pruning_threshold) {
        // Check connection count
        size_t connection_count = 0;
        auto out_it = outgoing_synapses_.find(neuron_id);
        auto in_it = incoming_synapses_.find(neuron_id);
        
        if (out_it != outgoing_synapses_.end()) connection_count += out_it->second.size();
        if (in_it != incoming_synapses_.end()) connection_count += in_it->second.size();
        
        return connection_count < 2; // Prune if very few connections
    }
    
    return false;
}

bool Network::shouldPruneSynapse(const Synapse& synapse) const {
    // Check weight threshold
    if (synapse.weight < config_.synapse_pruning_threshold) {
        return true;
    }
    
    // Check activity level
    if (synapse.activity_metric < 0.01) {
        return true;
    }
    
    // Check time since last potentiation
    double time_since_potentiation = current_time_ - synapse.last_potentiation;
    if (time_since_potentiation > 1000.0) { // 1 second
        return true;
    }
    
    return false;
}

std::vector<double> Network::calculateRegionalActivityLevels() const {
    return getRegionalActivity(5, 5, 100.0); // Use default 5x5 grid
}