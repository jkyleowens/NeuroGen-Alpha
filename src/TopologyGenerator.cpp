#include "TopologyGenerator.h"
#include <algorithm>
#include <climits>
#include <numeric>
#include <iostream>
#include <stdexcept>

TopologyGenerator::TopologyGenerator(const NetworkConfig& cfg)
    : cfg_(cfg), rng_(std::random_device{}()) {
    if (!cfg_.validate()) {
        throw std::runtime_error("Invalid network configuration passed to TopologyGenerator");
    }
}

void TopologyGenerator::buildLocalLoops(std::vector<GPUSynapse>& synapses,
                                       const std::vector<GPUCorticalColumn>& columns) {
    std::uniform_real_distribution<float> w_exc(cfg_.wExcMin, cfg_.wExcMax);
    std::uniform_real_distribution<float> w_inh(cfg_.wInhMin, cfg_.wInhMax);
    std::uniform_real_distribution<float> delay(cfg_.dMin, cfg_.dMax);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    synapses.clear();
    synapses.reserve(cfg_.totalSynapses);

    const int fanOut = cfg_.localFanOut;
    const int colSize = cfg_.neuronsPerColumn;
    const int excCut = static_cast<int>(cfg_.exc_ratio * colSize);

    for (const auto& col : columns) {
        for (int n = col.neuron_start; n < col.neuron_end; ++n) {
            // Classify neuron: first excCut neurons are excitatory
            bool isExc = (n - col.neuron_start) < excCut;

            // Create fanOut random connections within the same column
            for (int k = 0; k < fanOut; ++k) {
                int tgt = col.neuron_start + 
                         static_cast<int>(uni(rng_) * colSize);

                // Ensure we don't create self-connections
                if (tgt == n) {
                    tgt = (tgt + 1) % colSize + col.neuron_start;
                }

                GPUSynapse syn;
                syn.pre_neuron_idx = n;
                syn.post_neuron_idx = tgt;
                syn.delay = delay(rng_);
                syn.weight = isExc ? w_exc(rng_) : -w_inh(rng_);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;

                synapses.push_back(syn);
            }
        }
    }

    // Pad with dummy synapses if we're short (handles rounding errors)
    while (synapses.size() < cfg_.totalSynapses) {
        GPUSynapse dummy;
        dummy.pre_neuron_idx = 0;
        dummy.post_neuron_idx = 0;
        dummy.weight = 0.0f;
        dummy.delay = 1.0f;
        dummy.last_pre_spike_time = -1000.0f;
        dummy.activity_metric = 0.0f;
        synapses.emplace_back(dummy);
    }

    // Trim if we went over
    if (synapses.size() > cfg_.totalSynapses) {
        synapses.resize(cfg_.totalSynapses);
    }
}

void TopologyGenerator::buildInterColumnConnections(std::vector<GPUSynapse>& synapses,
                                                   const std::vector<GPUCorticalColumn>& columns,
                                                   float connection_probability) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(cfg_.wExcMin * 0.5f, cfg_.wExcMax * 0.5f);
    std::uniform_real_distribution<float> delay_dist(cfg_.dMin * 2.0f, cfg_.dMax * 3.0f);

    size_t initial_size = synapses.size();

    for (size_t i = 0; i < columns.size(); ++i) {
        for (size_t j = 0; j < columns.size(); ++j) {
            if (i == j) continue; // Skip intra-column connections

            const auto& source_col = columns[i];
            const auto& target_col = columns[j];

            // Create sparse inter-column connections
            for (int pre = source_col.neuron_start; pre < source_col.neuron_end; ++pre) {
                if (prob_dist(rng_) < connection_probability) {
                    // Select random target in destination column
                    int target_range = target_col.neuron_end - target_col.neuron_start;
                    int post = target_col.neuron_start + 
                              static_cast<int>(prob_dist(rng_) * target_range);

                    GPUSynapse syn;
                    syn.pre_neuron_idx = pre;
                    syn.post_neuron_idx = post;
                    syn.weight = weight_dist(rng_);
                    syn.delay = delay_dist(rng_);
                    syn.last_pre_spike_time = -1000.0f;
                    syn.activity_metric = 0.0f;

                    synapses.push_back(syn);
                }
            }
        }
    }

    std::cout << "[TOPOLOGY] Added " << (synapses.size() - initial_size) 
              << " inter-column connections" << std::endl;
}

void TopologyGenerator::buildInputConnections(std::vector<GPUSynapse>& synapses,
                                             int input_start, int input_end,
                                             int target_start, int target_end,
                                             float connection_probability) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(cfg_.wExcMin, cfg_.wExcMax);
    std::uniform_real_distribution<float> delay_dist(cfg_.dMin, cfg_.dMax);

    size_t initial_size = synapses.size();

    for (int pre = input_start; pre < input_end; ++pre) {
        for (int post = target_start; post < target_end; ++post) {
            if (prob_dist(rng_) < connection_probability) {
                GPUSynapse syn;
                syn.pre_neuron_idx = pre;
                syn.post_neuron_idx = post;
                syn.weight = weight_dist(rng_);
                syn.delay = delay_dist(rng_);
                syn.last_pre_spike_time = -1000.0f;
                syn.activity_metric = 0.0f;

                synapses.push_back(syn);
            }
        }
    }

    std::cout << "[TOPOLOGY] Added " << (synapses.size() - initial_size) 
              << " input connections" << std::endl;
}

void TopologyGenerator::shuffleConnections(std::vector<GPUSynapse>& synapses) {
    std::shuffle(synapses.begin(), synapses.end(), rng_);
}

void TopologyGenerator::validateTopology(const std::vector<GPUSynapse>& synapses,
                                        const std::vector<GPUCorticalColumn>& columns) const {
    if (synapses.empty()) {
        throw std::runtime_error("Empty synapse array");
    }

    // Find total neuron range
    int min_neuron = INT_MAX, max_neuron = INT_MIN;
    for (const auto& col : columns) {
        min_neuron = std::min(min_neuron, col.neuron_start);
        max_neuron = std::max(max_neuron, col.neuron_end - 1);
    }

    // Validate all synapse indices
    for (const auto& syn : synapses) {
        if (syn.pre_neuron_idx < min_neuron || syn.pre_neuron_idx > max_neuron ||
            syn.post_neuron_idx < min_neuron || syn.post_neuron_idx > max_neuron) {
            throw std::runtime_error("Synapse indices out of valid neuron range");
        }
        
        if (syn.delay < 0.0f || syn.delay > 100.0f) {
            throw std::runtime_error("Invalid synaptic delay");
        }
        
        if (std::abs(syn.weight) > 10.0f) {
            throw std::runtime_error("Synaptic weight out of reasonable range");
        }
    }

    std::cout << "[TOPOLOGY] Validation passed for " << synapses.size() 
              << " synapses" << std::endl;
}

void TopologyGenerator::printTopologyStats(const std::vector<GPUSynapse>& synapses,
                                          const std::vector<GPUCorticalColumn>& columns) const {
    if (synapses.empty()) return;

    float total_weight = 0.0f;
    float total_delay = 0.0f;
    int excitatory_count = 0;
    int inhibitory_count = 0;

    for (const auto& syn : synapses) {
        total_weight += std::abs(syn.weight);
        total_delay += syn.delay;
        
        if (syn.weight > 0) excitatory_count++;
        else if (syn.weight < 0) inhibitory_count++;
    }

    std::cout << "\n=== Topology Statistics ===" << std::endl;
    std::cout << "Total Synapses: " << synapses.size() << std::endl;
    std::cout << "Excitatory: " << excitatory_count << " (" 
              << (100.0f * excitatory_count / synapses.size()) << "%)" << std::endl;
    std::cout << "Inhibitory: " << inhibitory_count << " (" 
              << (100.0f * inhibitory_count / synapses.size()) << "%)" << std::endl;
    std::cout << "Average Weight: " << (total_weight / synapses.size()) << std::endl;
    std::cout << "Average Delay: " << (total_delay / synapses.size()) << " ms" << std::endl;
    std::cout << "Columns: " << columns.size() << std::endl;
    std::cout << "=========================" << std::endl;
}

// Private helper functions
bool TopologyGenerator::isExcitatoryNeuron(int neuron_idx, const GPUCorticalColumn& column) const {
    int relative_idx = neuron_idx - column.neuron_start;
    int excitatory_count = static_cast<int>(cfg_.exc_ratio * cfg_.neuronsPerColumn);
    return relative_idx < excitatory_count;
}

float TopologyGenerator::generateSynapticWeight(bool is_excitatory) const {
    std::uniform_real_distribution<float> exc_dist(cfg_.wExcMin, cfg_.wExcMax);
    std::uniform_real_distribution<float> inh_dist(cfg_.wInhMin, cfg_.wInhMax);
    
    if (is_excitatory) {
        return exc_dist(rng_);
    } else {
        return -inh_dist(rng_); // Negative for inhibitory
    }
}

float TopologyGenerator::generateSynapticDelay() const {
    std::uniform_real_distribution<float> delay_dist(cfg_.dMin, cfg_.dMax);
    return delay_dist(rng_);
}

int TopologyGenerator::selectRandomTarget(const GPUCorticalColumn& column) const {
    std::uniform_int_distribution<int> target_dist(column.neuron_start, column.neuron_end - 1);
    return target_dist(rng_);
}

// Configuration presets
namespace ConfigPresets {
    NetworkConfig smallNetwork() {
        NetworkConfig config;
        config.numColumns = 2;
        config.neuronsPerColumn = 128;
        config.localFanIn = 20;
        config.localFanOut = 20;
        config.finalizeConfig();
        return config;
    }

    NetworkConfig mediumNetwork() {
        NetworkConfig config;
        config.numColumns = 4;
        config.neuronsPerColumn = 256;
        config.localFanIn = 30;
        config.localFanOut = 30;
        config.finalizeConfig();
        return config;
    }

    NetworkConfig largeNetwork() {
        NetworkConfig config;
        config.numColumns = 8;
        config.neuronsPerColumn = 512;
        config.localFanIn = 40;
        config.localFanOut = 40;
        config.finalizeConfig();
        return config;
    }

    NetworkConfig testNetwork() {
        NetworkConfig config;
        config.numColumns = 1;
        config.neuronsPerColumn = 64;
        config.localFanIn = 10;
        config.localFanOut = 10;
        config.dt = 0.1f; // Larger time step for faster testing
        config.finalizeConfig();
        return config;
    }
}