#include <NeuroGen/TopologyGenerator.h>
#include <iostream>
#include <cmath>

TopologyGenerator::TopologyGenerator(const NetworkConfig& config)
    : config_(config), rng_(std::random_device{}()) {}

std::vector<GPUSynapse> TopologyGenerator::generate(const std::map<std::string, std::vector<NeuronModel>>& populations, const std::vector<ConnectionRule>& rules) 
{
    
    std::vector<GPUSynapse> all_synapses;
    std::vector<NeuronModel> all_neurons;
    for(const auto& pair : populations) {
        all_neurons.insert(all_neurons.end(), pair.second.begin(), pair.second.end());
    }

    // Initialize spatial grid for distance-based rules
    initializeSpatialGrid(all_neurons);

    for (const auto& rule : rules) {
        std::cout << "==> Applying Connection Rule: " << rule.source_pop_key << " -> " << rule.target_pop_key << std::endl;

        const auto& source_pop = populations.at(rule.source_pop_key);
        const auto& target_pop = populations.at(rule.target_pop_key);
        
        switch (rule.type) {
            case ConnectionRule::Type::DISTANCE_DECAY:
                applyDistanceDecayRule(rule, source_pop, target_pop, all_synapses);
                break;
            case ConnectionRule::Type::PROBABILISTIC:
                applyProbabilisticRule(rule, source_pop, target_pop, all_synapses);
                break;
            // Cases for other rule types would go here
            default:
                std::cerr << "Warning: ConnectionRule type not yet implemented." << std::endl;
                break;
        }
    }
    return all_synapses;
}

void TopologyGenerator::applyDistanceDecayRule(const ConnectionRule& rule, const std::vector<NeuronModel>& source_pop, const std::vector<NeuronModel>& target_pop, std::vector<GPUSynapse>& synapses) 
{

    std::normal_distribution<float> weight_dist(rule.weight_mean, rule.weight_std_dev);
    std::uniform_real_distribution<float> delay_dist(rule.delay_min, rule.delay_max);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (const auto& source_neuron : source_pop) {
        // EFFICIENTLY get nearby neurons using the spatial grid
        std::vector<int> nearby_indices = getNearbyNeurons(source_neuron, rule.max_distance);

        for (int target_id : nearby_indices) {
            const auto& target_neuron = grid_neurons_[target_id];
            
            // Ensure target is in the correct population
            if (target_neuron.population_key != rule.target_pop_key) continue;
            if (source_neuron.id == target_neuron.id) continue;

            float dist = std::sqrt(pow(source_neuron.x - target_neuron.x, 2) +
                                   pow(source_neuron.y - target_neuron.y, 2) +
                                   pow(source_neuron.z - target_neuron.z, 2));

            // Connection probability decays exponentially with distance
            float connection_prob = rule.probability * expf(-dist / rule.decay_constant);

            if (prob_dist(rng_) < connection_prob) {
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule.receptor_type, weight_dist(rng_), delay_dist(rng_)));
            }
        }
    }
}

void TopologyGenerator::applyProbabilisticRule(const ConnectionRule& rule, const std::vector<NeuronModel>& source_pop, const std::vector<NeuronModel>& target_pop, std::vector<GPUSynapse>& synapses) 
{
    std::normal_distribution<float> weight_dist(rule.weight_mean, rule.weight_std_dev);
    std::uniform_real_distribution<float> delay_dist(rule.delay_min, rule.delay_max);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    // This is the inefficient O(N*M) loop we are trying to move away from,
    // but it is kept for purely probabilistic rules without spatial constraints.
    for (const auto& source_neuron : source_pop) {
        for (const auto& target_neuron : target_pop) {
            if (source_neuron.id == target_neuron.id) continue;

            if (prob_dist(rng_) < rule.probability) {
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule.receptor_type, weight_dist(rng_), delay_dist(rng_)));
            }
        }
    }
}


// --- Spatial Grid Implementation for Efficiency ---

void TopologyGenerator::initializeSpatialGrid(const std::vector<NeuronModel>& all_neurons) 
{
    grid_neurons_ = all_neurons;
    float max_x = 0, max_y = 0, max_z = 0;
    for(const auto& n : all_neurons) {
        if(n.x > max_x) max_x = n.x;
        if(n.y > max_y) max_y = n.y;
        if(n.z > max_z) max_z = n.z;
    }

    int x_bins = static_cast<int>(max_x / grid_bin_size_) + 1;
    int y_bins = static_cast<int>(max_y / grid_bin_size_) + 1;
    int z_bins = static_cast<int>(max_z / grid_bin_size_) + 1;

    spatial_grid_.assign(x_bins, std::vector<std::vector<std::vector<int>>>(y_bins, std::vector<std::vector<int>>(z_bins)));

    for (const auto& n : all_neurons) {
        int x_idx = static_cast<int>(n.x / grid_bin_size_);
        int y_idx = static_cast<int>(n.y / grid_bin_size_);
        int z_idx = static_cast<int>(n.z / grid_bin_size_);
        spatial_grid_[x_idx][y_idx][z_idx].push_back(n.id);
    }
}

std::vector<int> TopologyGenerator::getNearbyNeurons(const NeuronModel& neuron, float radius) 
{
    std::vector<int> nearby_neurons;
    int x_bins = spatial_grid_.size();
    if (x_bins == 0) return nearby_neurons;
    int y_bins = spatial_grid_[0].size();
    if (y_bins == 0) return nearby_neurons;
    int z_bins = spatial_grid_[0][0].size();
    if (z_bins == 0) return nearby_neurons;

    int center_x_idx = static_cast<int>(neuron.x / grid_bin_size_);
    int center_y_idx = static_cast<int>(neuron.y / grid_bin_size_);
    int center_z_idx = static_cast<int>(neuron.z / grid_bin_size_);
    int search_range = static_cast<int>(radius / grid_bin_size_) + 1;

    for (int x = -search_range; x <= search_range; ++x) {
        for (int y = -search_range; y <= search_range; ++y) {
            for (int z = -search_range; z <= search_range; ++z) {
                int current_x = center_x_idx + x;
                int current_y = center_y_idx + y;
                int current_z = center_z_idx + z;

                if (current_x >= 0 && current_x < x_bins &&
                    current_y >= 0 && current_y < y_bins &&
                    current_z >= 0 && current_z < z_bins) {
                    nearby_neurons.insert(nearby_neurons.end(), 
                                          spatial_grid_[current_x][current_y][current_z].begin(),
                                          spatial_grid_[current_x][current_y][current_z].end());
                }
            }
        }
    }
    return nearby_neurons;
}

GPUSynapse TopologyGenerator::createSynapse(int pre_id, int post_id, int receptor, float weight, float delay) 
{
    GPUSynapse syn{};
    syn.pre_neuron_idx = pre_id;
    syn.post_neuron_idx = post_id;
    syn.weight = weight;
    syn.delay = delay;
    syn.receptor_index = receptor;
    syn.active = 1;
    // Initialize other plasticity-related fields to defaults
    syn.eligibility_trace = 0.0f;
    syn.dopamine_sensitivity = 1.0f;
    syn.plasticity_modulation = 1.0f;
    return syn;
}