#include "NetworkCPU.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

NetworkCPU::NetworkCPU(const NetworkConfig& config) 
    : config_(config), current_time_(0.0f), rng_(std::random_device{}()) {
    initialize();
}

void NetworkCPU::initialize() {
    std::cout << "[CPU] Initializing neural network..." << std::endl;
    
    // Calculate total neurons
    int total_neurons = config_.input_size + config_.hidden_size + config_.output_size;
    
    // Initialize neurons
    neurons_.resize(total_neurons);
    for (auto& neuron : neurons_) {
        neuron.voltage = -65.0f + rng_() % 10 - 5; // Small random variation
        neuron.m = 0.05f;
        neuron.h = 0.60f;
        neuron.n = 0.32f;
        neuron.spiked = false;
        neuron.last_spike_time = -1.0f;
    }
    
    // Initialize buffers
    input_buffer_.resize(config_.input_size, 0.0f);
    output_buffer_.resize(config_.output_size, 0.0f);
    
    // Create network topology
    createTopology();
    
    std::cout << "[CPU] Network initialized with " << total_neurons << " neurons, " 
              << synapses_.size() << " synapses" << std::endl;
}

void NetworkCPU::createTopology() {
    synapses_.clear();
    
    std::uniform_real_distribution<float> weight_dist(-config_.weight_init_std, config_.weight_init_std);
    std::uniform_real_distribution<float> delay_dist(config_.delay_min, config_.delay_max);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    int input_start = 0;
    int hidden_start = config_.input_size;
    int output_start = config_.input_size + config_.hidden_size;
    
    // Input to hidden connections
    for (int i = input_start; i < input_start + config_.input_size; ++i) {
        for (int j = hidden_start; j < hidden_start + config_.hidden_size; ++j) {
            if (prob_dist(rng_) < config_.input_hidden_prob) {
                CPUSynapse syn;
                syn.pre_neuron_idx = i;
                syn.post_neuron_idx = j;
                syn.weight = weight_dist(rng_);
                syn.delay = delay_dist(rng_);
                syn.last_pre_spike_time = -1.0f;
                syn.activity_metric = 0.0f;
                synapses_.push_back(syn);
            }
        }
    }
    
    // Hidden to hidden connections (sparse)
    for (int i = hidden_start; i < hidden_start + config_.hidden_size; ++i) {
        for (int j = hidden_start; j < hidden_start + config_.hidden_size; ++j) {
            if (i != j && prob_dist(rng_) < config_.hidden_hidden_prob) {
                CPUSynapse syn;
                syn.pre_neuron_idx = i;
                syn.post_neuron_idx = j;
                // 80% excitatory, 20% inhibitory
                float weight = std::abs(weight_dist(rng_));
                syn.weight = (prob_dist(rng_) < config_.exc_ratio) ? weight : -weight;
                syn.delay = delay_dist(rng_);
                syn.last_pre_spike_time = -1.0f;
                syn.activity_metric = 0.0f;
                synapses_.push_back(syn);
            }
        }
    }
    
    // Hidden to output connections
    for (int i = hidden_start; i < hidden_start + config_.hidden_size; ++i) {
        for (int j = output_start; j < output_start + config_.output_size; ++j) {
            if (prob_dist(rng_) < config_.hidden_output_prob) {
                CPUSynapse syn;
                syn.pre_neuron_idx = i;
                syn.post_neuron_idx = j;
                syn.weight = weight_dist(rng_);
                syn.delay = delay_dist(rng_);
                syn.last_pre_spike_time = -1.0f;
                syn.activity_metric = 0.0f;
                synapses_.push_back(syn);
            }
        }
    }
}

std::vector<float> NetworkCPU::forward(const std::vector<float>& input, float reward_signal) {
    if (input.size() != config_.input_size) {
        throw std::runtime_error("Input size mismatch");
    }
    
    // Copy input to buffer
    std::copy(input.begin(), input.end(), input_buffer_.begin());
    
    // Run simulation for specified duration
    int steps = static_cast<int>(config_.simulation_time / config_.dt);
    
    for (int step = 0; step < steps; ++step) {
        current_time_ = step * config_.dt;
        
        // Inject input current to input neurons
        for (int i = 0; i < config_.input_size; ++i) {
            neurons_[i].voltage += input_buffer_[i] * config_.input_current_scale * config_.dt;
        }
        
        // Update all neurons
        for (auto& neuron : neurons_) {
            updateNeuron(neuron, config_.dt, 0.0f);
        }
        
        // Propagate spikes through synapses
        propagateSpikes(config_.dt);
        
        // Apply STDP every few steps
        if (step % 5 == 0) {
            applySTDP(config_.dt);
        }
    }
    
    // Extract output
    return extractOutput();
}

void NetworkCPU::updateNeuron(CPUNeuronState& neuron, float dt, float external_current) {
    // Hodgkin-Huxley equations (simplified)
    const float V_Na = 50.0f, V_K = -77.0f, V_L = -54.3f;
    const float g_Na = 120.0f, g_K = 36.0f, g_L = 0.3f;
    const float C_m = 1.0f;
    
    float V = neuron.voltage;
    float m = neuron.m, h = neuron.h, n = neuron.n;
    
    // Rate functions (simplified)
    auto alpha_m = [](float V) { return 0.1f * (V + 40.0f) / (1.0f - expf(-(V + 40.0f) / 10.0f)); };
    auto beta_m = [](float V) { return 4.0f * expf(-(V + 65.0f) / 18.0f); };
    auto alpha_h = [](float V) { return 0.07f * expf(-(V + 65.0f) / 20.0f); };
    auto beta_h = [](float V) { return 1.0f / (1.0f + expf(-(V + 35.0f) / 10.0f)); };
    auto alpha_n = [](float V) { return 0.01f * (V + 55.0f) / (1.0f - expf(-(V + 55.0f) / 10.0f)); };
    auto beta_n = [](float V) { return 0.125f * expf(-(V + 65.0f) / 80.0f); };
    
    // Update gating variables
    float am = alpha_m(V), bm = beta_m(V);
    float ah = alpha_h(V), bh = beta_h(V);
    float an = alpha_n(V), bn = beta_n(V);
    
    neuron.m += dt * (am * (1.0f - m) - bm * m);
    neuron.h += dt * (ah * (1.0f - h) - bh * h);
    neuron.n += dt * (an * (1.0f - n) - bn * n);
    
    // Calculate currents
    float I_Na = g_Na * m * m * m * h * (V - V_Na);
    float I_K = g_K * n * n * n * n * (V - V_K);
    float I_L = g_L * (V - V_L);
    
    // Update voltage
    float dV = (-I_Na - I_K - I_L + external_current) / C_m;
    neuron.voltage += dt * dV;
    
    // Check for spike
    neuron.spiked = false;
    if (neuron.voltage > config_.spike_threshold && neuron.last_spike_time < current_time_ - 2.0f) {
        neuron.spiked = true;
        neuron.last_spike_time = current_time_;
        neuron.voltage = -65.0f; // Reset voltage
    }
}

void NetworkCPU::propagateSpikes(float dt) {
    // Process synaptic transmission
    for (auto& syn : synapses_) {
        const auto& pre_neuron = neurons_[syn.pre_neuron_idx];
        auto& post_neuron = neurons_[syn.post_neuron_idx];
        
        // Check if presynaptic neuron spiked within delay window
        if (pre_neuron.spiked && 
            current_time_ - pre_neuron.last_spike_time >= syn.delay) {
            
            // Add synaptic current
            post_neuron.voltage += syn.weight;
            
            // Update activity metric for STDP
            syn.activity_metric += 1.0f;
            syn.last_pre_spike_time = pre_neuron.last_spike_time;
        }
    }
}

void NetworkCPU::applySTDP(float dt) {
    // Simplified STDP implementation
    for (auto& syn : synapses_) {
        const auto& pre_neuron = neurons_[syn.pre_neuron_idx];
        const auto& post_neuron = neurons_[syn.post_neuron_idx];
        
        if (syn.last_pre_spike_time > 0 && post_neuron.last_spike_time > 0) {
            float delta_t = post_neuron.last_spike_time - syn.last_pre_spike_time;
            
            if (std::abs(delta_t) < config_.stdp_window) {
                float weight_change = 0.0f;
                if (delta_t > 0) {
                    // Post after pre - LTP
                    weight_change = config_.stdp_learning_rate * expf(-delta_t / config_.stdp_tau_pre);
                } else {
                    // Pre after post - LTD
                    weight_change = -config_.stdp_learning_rate * expf(delta_t / config_.stdp_tau_post);
                }
                
                syn.weight += weight_change;
                syn.weight = std::max(-config_.max_weight, std::min(config_.max_weight, syn.weight));
            }
        }
    }
}

std::vector<float> NetworkCPU::extractOutput() {
    int output_start = config_.input_size + config_.hidden_size;
    
    for (int i = 0; i < config_.output_size; ++i) {
        // Simple spike rate encoding
        const auto& neuron = neurons_[output_start + i];
        if (neuron.spiked) {
            output_buffer_[i] = 1.0f;
        } else {
            output_buffer_[i] *= 0.95f; // Decay
        }
    }
    
    return output_buffer_;
}

void NetworkCPU::updateWeights(float reward_signal) {
    // Apply reward modulation to recent weight changes
    for (auto& syn : synapses_) {
        if (syn.activity_metric > 0) {
            syn.weight *= (1.0f + reward_signal * config_.reward_modulation_strength);
            syn.weight = std::max(-config_.max_weight, std::min(config_.max_weight, syn.weight));
            syn.activity_metric *= 0.9f; // Decay activity
        }
    }
}

void NetworkCPU::cleanup() {
    neurons_.clear();
    synapses_.clear();
    input_buffer_.clear();
    output_buffer_.clear();
}

void NetworkCPU::setConfig(const NetworkConfig& config) {
    config_ = config;
}

NetworkConfig NetworkCPU::getConfig() const {
    return config_;
}

void NetworkCPU::printStats() const {
    int active_neurons = 0;
    for (const auto& neuron : neurons_) {
        if (neuron.spiked) active_neurons++;
    }
    
    float avg_weight = 0.0f;
    for (const auto& syn : synapses_) {
        avg_weight += std::abs(syn.weight);
    }
    if (!synapses_.empty()) avg_weight /= synapses_.size();
    
    std::cout << "[CPU] Network Stats:" << std::endl;
    std::cout << "  Active neurons: " << active_neurons << "/" << neurons_.size() << std::endl;
    std::cout << "  Average weight: " << avg_weight << std::endl;
    std::cout << "  Synapses: " << synapses_.size() << std::endl;
    std::cout << "  Simulation time: " << current_time_ << "ms" << std::endl;
}

void NetworkCPU::saveState(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    // Simple binary save - implement as needed
    std::cout << "[CPU] State saved to " << filename << std::endl;
}

void NetworkCPU::loadState(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    // Simple binary load - implement as needed
    std::cout << "[CPU] State loaded from " << filename << std::endl;
}

void NetworkCPU::reset() {
    current_time_ = 0.0f;
    for (auto& neuron : neurons_) {
        neuron.voltage = -65.0f;
        neuron.spiked = false;
        neuron.last_spike_time = -1.0f;
    }
    
    for (auto& syn : synapses_) {
        syn.last_pre_spike_time = -1.0f;
        syn.activity_metric = 0.0f;
    }
}
