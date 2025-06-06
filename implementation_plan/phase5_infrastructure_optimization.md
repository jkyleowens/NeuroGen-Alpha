# Phase 5: Infrastructure Optimization

## Overview

The current implementation has inefficiencies in topology generation, data handling, and trading simulation. This phase will optimize these components to improve performance, scalability, and realism.

## Current Implementation Analysis

From the code analysis, we found:

- Topology generation uses a brute-force approach that doesn't scale well
- Data pipeline is limited to simple CSV files and lacks support for order book data
- Trading simulation lacks realistic execution modeling and risk management
- Performance metrics are limited to basic profit and loss

## Implementation Tasks

### 1. Optimize Topology Generation

#### 1.1 Implement Efficient Topology Generator

```cpp
// EnhancedTopologyGenerator.h
#ifndef ENHANCED_TOPOLOGY_GENERATOR_H
#define ENHANCED_TOPOLOGY_GENERATOR_H

#include <vector>
#include <random>
#include <unordered_map>
#include <functional>
#include "GPUNeuralStructures.h"

class EnhancedTopologyGenerator {
private:
    // Configuration
    NetworkConfig cfg_;
    
    // Random number generation
    std::mt19937 rng_;
    
    // Connection probability matrices
    std::vector<std::vector<float>> layer_connection_probabilities_;
    std::vector<std::vector<float>> column_connection_probabilities_;
    
    // Distance-dependent connection functions
    std::function<float(float)> excitatory_distance_function_;
    std::function<float(float)> inhibitory_distance_function_;
    
    // Spatial organization
    std::vector<std::tuple<float, float, float>> neuron_positions_;  // (x, y, z)
    
    // Helper methods
    float calculateDistance(int neuron1, int neuron2) const;
    bool isExcitatoryNeuron(int neuron_idx, const GPUCorticalColumn& column) const;
    float generateSynapticWeight(bool is_excitatory, int source_layer, int target_layer) const;
    float generateSynapticDelay(float distance) const;
    int selectTargetCompartment(int target_neuron, bool is_excitatory) const;
    int selectReceptorType(bool is_excitatory, int target_compartment) const;
    
public:
    EnhancedTopologyGenerator(const NetworkConfig& cfg);
    
    // Initialization
    void initializeConnectionProbabilities();
    void generateNeuronPositions();
    
    // Topology generation
    void buildLocalConnections(std::vector<GPUSynapse>& synapses,
                              const std::vector<GPUCorticalColumn>& columns);
    void buildInterColumnConnections(std::vector<GPUSynapse>& synapses,
                                    const std::vector<GPUCorticalColumn>& columns);
    void buildLayerSpecificConnections(std::vector<GPUSynapse>& synapses,
                                      const std::vector<GPUCorticalColumn>& columns);
    void buildDistanceDependentConnections(std::vector<GPUSynapse>& synapses,
                                          const std::vector<GPUCorticalColumn>& columns);
    
    // Utility functions
    void validateTopology(const std::vector<GPUSynapse>& synapses,
                         const std::vector<GPUCorticalColumn>& columns) const;
    void printTopologyStats(const std::vector<GPUSynapse>& synapses,
                           const std::vector<GPUCorticalColumn>& columns) const;
    void optimizeMemoryLayout(std::vector<GPUSynapse>& synapses);
};

#endif // ENHANCED_TOPOLOGY_GENERATOR_H
```

#### 1.2 Implement Layer-Specific Connection Probabilities

```cpp
void EnhancedTopologyGenerator::initializeConnectionProbabilities() {
    // Initialize layer connection probability matrix
    // This defines the probability of connection between neurons in different layers
    // Format: [source_layer][target_layer] = probability
    
    const int num_layers = cfg_.numLayers;
    layer_connection_probabilities_.resize(num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        layer_connection_probabilities_[i].resize(num_layers, 0.0f);
    }
    
    // Set connection probabilities based on biological cortical connectivity
    // Layer 1 (input)
    layer_connection_probabilities_[0][0] = 0.05f;  // Recurrent connections within input layer
    layer_connection_probabilities_[0][1] = 0.15f;  // Input -> Layer 2/3
    layer_connection_probabilities_[0][2] = 0.05f;  // Input -> Layer 4
    layer_connection_probabilities_[0][3] = 0.05f;  // Input -> Layer 5
    layer_connection_probabilities_[0][4] = 0.01f;  // Input -> Layer 6
    
    // Layer 2/3 (superficial pyramidal)
    layer_connection_probabilities_[1][0] = 0.05f;  // Layer 2/3 -> Input
    layer_connection_probabilities_[1][1] = 0.15f;  // Recurrent within Layer 2/3
    layer_connection_probabilities_[1][2] = 0.08f;  // Layer 2/3 -> Layer 4
    layer_connection_probabilities_[1][3] = 0.1f;   // Layer 2/3 -> Layer 5
    layer_connection_probabilities_[1][4] = 0.05f;  // Layer 2/3 -> Layer 6
    
    // Layer 4 (granular)
    layer_connection_probabilities_[2][0] = 0.02f;  // Layer 4 -> Input
    layer_connection_probabilities_[2][1] = 0.2f;   // Layer 4 -> Layer 2/3 (strong)
    layer_connection_probabilities_[2][2] = 0.1f;   // Recurrent within Layer 4
    layer_connection_probabilities_[2][3] = 0.05f;  // Layer 4 -> Layer 5
    layer_connection_probabilities_[2][4] = 0.05f;  // Layer 4 -> Layer 6
    
    // Layer 5 (deep pyramidal)
    layer_connection_probabilities_[3][0] = 0.03f;  // Layer 5 -> Input
    layer_connection_probabilities_[3][1] = 0.05f;  // Layer 5 -> Layer 2/3
    layer_connection_probabilities_[3][2] = 0.05f;  // Layer 5 -> Layer 4
    layer_connection_probabilities_[3][3] = 0.15f;  // Recurrent within Layer 5
    layer_connection_probabilities_[3][4] = 0.1f;   // Layer 5 -> Layer 6
    
    // Layer 6 (output)
    layer_connection_probabilities_[4][0] = 0.01f;  // Layer 6 -> Input
    layer_connection_probabilities_[4][1] = 0.05f;  // Layer 6 -> Layer 2/3
    layer_connection_probabilities_[4][2] = 0.1f;   // Layer 6 -> Layer 4
    layer_connection_probabilities_[4][3] = 0.05f;  // Layer 6 -> Layer 5
    layer_connection_probabilities_[4][4] = 0.1f;   // Recurrent within Layer 6
    
    // Initialize column connection probability matrix
    // This defines the probability of connection between neurons in different columns
    const int num_columns = cfg_.numColumns;
    column_connection_probabilities_.resize(num_columns);
    
    for (int i = 0; i < num_columns; i++) {
        column_connection_probabilities_[i].resize(num_columns, 0.0f);
    }
    
    // Set default inter-column connection probability
    float base_probability = 0.05f;
    
    // Modify based on column function and distance
    for (int i = 0; i < num_columns; i++) {
        for (int j = 0; j < num_columns; j++) {
            if (i == j) {
                // No self-connections at column level (handled by intra-column connections)
                column_connection_probabilities_[i][j] = 0.0f;
            } else {
                // Base probability modified by column type and distance
                // This will be further refined in generateNeuronPositions()
                column_connection_probabilities_[i][j] = base_probability;
            }
        }
    }
    
    // Initialize distance-dependent connection functions
    
    // Excitatory neurons follow a Gaussian-like connectivity pattern
    // Higher probability of connecting to nearby neurons, dropping off with distance
    excitatory_distance_function_ = [](float distance) -> float {
        const float sigma = 0.2f;  // Spatial scale
        return expf(-(distance * distance) / (2.0f * sigma * sigma));
    };
    
    // Inhibitory neurons have more uniform connectivity, with a slight distance dependence
    inhibitory_distance_function_ = [](float distance) -> float {
        const float sigma = 0.5f;  // Larger spatial scale than excitatory
        return 0.5f + 0.5f * expf(-(distance * distance) / (2.0f * sigma * sigma));
    };
}
```

#### 1.3 Implement Distance-Dependent Connectivity

```cpp
void EnhancedTopologyGenerator::generateNeuronPositions() {
    // Generate 3D positions for all neurons
    const int total_neurons = cfg_.totalNeurons;
    neuron_positions_.resize(total_neurons);
    
    // For each column, generate positions for its neurons
    int neuron_idx = 0;
    for (int col = 0; col < cfg_.numColumns; col++) {
        // Column center position (in normalized 0-1 space)
        float col_x = static_cast<float>(col % static_cast<int>(sqrtf(cfg_.numColumns))) / 
                     sqrtf(cfg_.numColumns);
        float col_y = static_cast<float>(col / static_cast<int>(sqrtf(cfg_.numColumns))) / 
                     sqrtf(cfg_.numColumns);
        float col_z = 0.5f;  // All columns at same z-level initially
        
        // Column radius (normalized)
        float col_radius = 0.5f / sqrtf(cfg_.numColumns);
        
        // Generate positions for neurons in this column
        for (int n = 0; n < cfg_.neuronsPerColumn; n++) {
            // Determine neuron's layer
            int layer = n / (cfg_.neuronsPerColumn / cfg_.numLayers);
            
            // Layer-specific z-coordinate
            float z = static_cast<float>(layer) / cfg_.numLayers;
            
            // Random position within column radius (2D disc)
            float radius = col_radius * sqrtf(std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_));
            float angle = std::uniform_real_distribution<float>(0.0f, 2.0f * 3.14159f)(rng_);
            
            float x = col_x + radius * cosf(angle);
            float y = col_y + radius * sinf(angle);
            
            // Store position
            neuron_positions_[neuron_idx] = std::make_tuple(x, y, z);
            neuron_idx++;
        }
    }
    
    // Update column connection probabilities based on distances
    for (int i = 0; i < cfg_.numColumns; i++) {
        int i_start = i * cfg_.neuronsPerColumn;
        
        for (int j = 0; j < cfg_.numColumns; j++) {
            if (i == j) continue;  // Skip self
            
            int j_start = j * cfg_.neuronsPerColumn;
            
            // Calculate average distance between columns
            float total_distance = 0.0f;
            int count = 0;
            
            // Sample a subset of neurons for efficiency
            for (int i_sample = 0; i_sample < 10; i_sample++) {
                int i_idx = i_start + i_sample * (cfg_.neuronsPerColumn / 10);
                
                for (int j_sample = 0; j_sample < 10; j_sample++) {
                    int j_idx = j_start + j_sample * (cfg_.neuronsPerColumn / 10);
                    
                    total_distance += calculateDistance(i_idx, j_idx);
                    count++;
                }
            }
            
            float avg_distance = total_distance / count;
            
            // Update connection probability based on distance
            // Closer columns have higher connection probability
            column_connection_probabilities_[i][j] = 0.05f * expf(-avg_distance * 5.0f);
        }
    }
}

float EnhancedTopologyGenerator::calculateDistance(int neuron1, int neuron2) const {
    // Calculate Euclidean distance between two neurons
    float x1, y1, z1, x2, y2, z2;
    std::tie(x1, y1, z1) = neuron_positions_[neuron1];
    std::tie(x2, y2, z2) = neuron_positions_[neuron2];
    
    return sqrtf((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
}
```

#### 1.4 Implement Efficient Connection Generation

```cpp
void EnhancedTopologyGenerator::buildDistanceDependentConnections(
    std::vector<GPUSynapse>& synapses,
    const std::vector<GPUCorticalColumn>& columns) {
    
    // Reserve space for synapses
    size_t estimated_synapses = cfg_.totalNeurons * cfg_.avgSynapsesPerNeuron;
    synapses.reserve(estimated_synapses);
    
    // Create spatial index for efficient neighbor finding
    // This is a simple grid-based spatial partitioning
    const int grid_size = 10;  // 10x10x10 grid
    std::vector<std::vector<int>> spatial_grid(grid_size * grid_size * grid_size);
    
    // Assign neurons to grid cells
    for (int n = 0; n < cfg_.totalNeurons; n++) {
        float x, y, z;
        std::tie(x, y, z) = neuron_positions_[n];
        
        // Convert to grid coordinates
        int gx = static_cast<int>(x * grid_size);
        int gy = static_cast<int>(y * grid_size);
        int gz = static_cast<int>(z * grid_size);
        
        // Clamp to valid range
        gx = std::max(0, std::min(grid_size - 1, gx));
        gy = std::max(0, std::min(grid_size - 1, gy));
        gz = std::max(0, std::min(grid_size - 1, gz));
        
        // Add to grid
        int grid_idx = gz * grid_size * grid_size + gy * grid_size + gx;
        spatial_grid[grid_idx].push_back(n);
    }
    
    // For each neuron, create connections based on distance
    std::uniform_real_distribution<float> uni_dist(0.0f, 1.0f);
    
    for (int n = 0; n < cfg_.totalNeurons; n++) {
        // Determine neuron properties
        int column_idx = n / cfg_.neuronsPerColumn;
        int local_idx = n % cfg_.neuronsPerColumn;
        int layer = local_idx / (cfg_.neuronsPerColumn / cfg_.numLayers);
        bool is_excitatory = isExcitatoryNeuron(n, columns[column_idx]);
        
        // Get neuron position
        float nx, ny, nz;
        std::tie(nx, ny, nz) = neuron_positions_[n];
        
        // Convert to grid coordinates
        int gx = static_cast<int>(nx * grid_size);
        int gy = static_cast<int>(ny * grid_size);
        int gz = static_cast<int>(nz * grid_size);
        
        // Clamp to valid range
        gx = std::max(0, std::min(grid_size - 1, gx));
        gy = std::max(0, std::min(grid_size - 1, gy));
        gz = std::max(0, std::min(grid_size - 1, gz));
        
        // Search neighboring grid cells
        for (int dz = -1; dz <= 1; dz++) {
            int z = gz + dz;
            if (z < 0 || z >= grid_size) continue;
            
            for (int dy = -1; dy <= 1; dy++) {
                int y = gy + dy;
                if (y < 0 || y >= grid_size) continue;
                
                for (int dx = -1; dx <= 1; dx++) {
                    int x = gx + dx;
                    if (x < 0 || x >= grid_size) continue;
                    
                    // Get neurons in this grid cell
                    int grid_idx = z * grid_size * grid_size + y * grid_size + x;
                    const auto& cell_neurons = spatial_grid[grid_idx];
                    
                    // Check each potential target neuron
                    for (int target : cell_neurons) {
                        // Skip self-connections
                        if (target == n) continue;
                        
                        // Determine target properties
                        int target_column = target / cfg_.neuronsPerColumn;
                        int target_local = target % cfg_.neuronsPerColumn;
                        int target_layer = target_local / (cfg_.neuronsPerColumn / cfg_.numLayers);
                        
                        // Get base connection probability from layer and column matrices
                        float layer_prob = layer_connection_probabilities_[layer][target_layer];
                        float column_prob = column_connection_probabilities_[column_idx][target_column];
                        
                        // Calculate distance
                        float distance = calculateDistance(n, target);
                        
                        // Apply distance-dependent probability
                        float distance_factor = is_excitatory ? 
                            excitatory_distance_function_(distance) : 
                            inhibitory_distance_function_(distance);
                        
                        // Final connection probability
                        float connection_prob = layer_prob * column_prob * distance_factor;
                        
                        // Decide whether to create connection
                        if (uni_dist(rng_) < connection_prob) {
                            // Create synapse
                            GPUSynapse synapse;
                            synapse.pre_neuron_idx = n;
                            synapse.post_neuron_idx = target;
                            
                            // Determine target compartment and receptor type
                            synapse.post_compartment = selectTargetCompartment(target, is_excitatory);
                            synapse.receptor_index = selectReceptorType(is_excitatory, synapse.post_compartment);
                            
                            // Set weight based on pre/post layer and excitatory/inhibitory type
                            synapse.weight = generateSynapticWeight(is_excitatory, layer, target_layer);
                            
                            // Set delay based on distance
                            synapse.delay = generateSynapticDelay(distance);
                            
                            // Initialize other fields
                            synapse.last_pre_spike_time = -1000.0f;
                            synapse.activity_metric = 0.0f;
                            synapse.active = 1;
                            
                            // Add to synapse vector
                            synapses.push_back(synapse);
                        }
                    }
                }
            }
        }
    }
    
    // Ensure we have the required number of synapses
    std::cout << "Generated " << synapses.size() << " synapses using distance-dependent connectivity." << std::endl;
    
    // If we have too few synapses, add more random ones
    if (synapses.size() < cfg_.minTotalSynapses) {
        std::cout << "Adding " << (cfg_.minTotalSynapses - synapses.size()) 
                  << " additional random synapses to meet minimum requirement." << std::endl;
        
        // Add random synapses until we reach the minimum
        while (synapses.size() < cfg_.minTotalSynapses) {
            // Pick random pre and post neurons
            int pre = std::uniform_int_distribution<int>(0, cfg_.totalNeurons - 1)(rng_);
            int post = std::uniform_int_distribution<int>(0, cfg_.totalNeurons - 1)(rng_);
            
            // Skip self-connections
            if (pre == post) continue;
            
            // Create synapse
            GPUSynapse synapse;
            synapse.pre_neuron_idx = pre;
            synapse.post_neuron_idx = post;
            
            // Determine if pre neuron is excitatory
            int pre_column = pre / cfg_.neuronsPerColumn;
            bool is_excitatory = isExcitatoryNeuron(pre, columns[pre_column]);
            
            // Set other properties
            synapse.post_compartment = selectTargetCompartment(post, is_excitatory);
            synapse.receptor_index = selectReceptorType(is_excitatory, synapse.post_compartment);
            synapse.weight = is_excitatory ? 
                std::uniform_real_distribution<float>(cfg_.wExcMin, cfg_.wExcMax)(rng_) : 
                -std::uniform_real_distribution<float>(cfg_.wInhMin, cfg_.wInhMax)(rng_);
            synapse.delay = std::uniform_real_distribution<float>(cfg_.dMin, cfg_.dMax)(rng_);
            synapse.last_pre_spike_time = -1000.0f;
            synapse.activity_metric = 0.0f;
            synapse.active = 1;
            
            // Add to synapse vector
            synapses.push_back(synapse);
        }
    }
    
    // If we have too many synapses, remove some randomly
    if (synapses.size() > cfg_.maxTotalSynapses) {
        std::cout << "Removing " << (synapses.size() - cfg_.maxTotalSynapses) 
                  << " random synapses to meet maximum limit." << std::endl;
        
        // Shuffle and resize
        std::shuffle(synapses.begin(), synapses.end(), rng_);
        synapses.resize(cfg_.maxTotalSynapses);
    }
}
```

#### 1.5 Optimize Memory Layout

```cpp
void EnhancedTopologyGenerator::optimizeMemoryLayout(std::vector<GPUSynapse>& synapses) {
    // This function reorders synapses to improve memory access patterns on the GPU
    
    // Sort synapses by post-neuron index to improve memory coalescing
    // This helps when multiple threads access synapses targeting the same neuron
    std::sort(synapses.begin(), synapses.end(), 
             [](const GPUSynapse& a, const GPUSynapse& b) {
                 return a.post_neuron_idx < b.post_neuron_idx;
             });
    
    // Create index mapping for fast lookup of synapses by post-neuron
    std::vector<std::pair<int, int>> post_neuron_ranges(cfg_.totalNeurons, {-1, -1});
    
    int current_post = -1;
    int start_idx = -1;
    
    for (size_t i = 0; i < synapses.size(); i++) {
        int post = synapses[i].post_neuron_idx;
        
        if (post != current_post) {
            // End previous range
            if (current_post != -1) {
                post_neuron_ranges[current_post].second = i;
            }
            
            // Start new range
            current_post = post;
            start_idx = i;
            post_neuron_ranges[post].first = start_idx;
        }
    }
    
    // End the last range
    if (current_post != -1) {
        post_neuron_ranges[current_post].second = synapses.size();
    }
    
    // Store the index mapping for later use
    // This can be used by the network to quickly find all synapses targeting a specific neuron
    // (Implementation detail: this would be stored in the network object)
    
    std::cout << "Memory layout optimized for GPU access patterns." << std::endl;
}
```

### 2. Enhance Data Pipeline

#### 2.1 Create Advanced Market Data Processor

```cpp
// MarketDataProcessor.h
#ifndef MARKET_DATA_PROCESSOR_H
#define MARKET_DATA_PROCESSOR_H

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <functional>

// Forward declarations
class OrderBookSnapshot;
class MarketTrade;
class NewsItem;
class SentimentData;

// Market data feature vector
struct MarketFeatures {
    // Price features
    float open;
    float high;
    float low;
    float close;
    float volume;
    
    // Technical indicators
    float sma_5;
    float sma_20;
    float ema_5;
    float ema_20;
    float rsi_14;
    float macd;
    float macd_signal;
    float bollinger_upper;
    float bollinger_lower;
    float atr_14;
    
    // Order book features
    float bid_ask_spread;
    float bid_ask_imbalance;
    float order_book_pressure;
    float depth_1_price_delta;
    float depth_2_price_delta;
    float depth_3_price_delta;
    float depth_1_volume_ratio;
    float depth_2_volume_ratio;
    float depth_3_volume_ratio;
    
    // Sentiment features
    float news_sentiment;
    float social_sentiment;
    float sentiment_change_rate;
    
    // Volatility features
    float realized_volatility;
    float implied_volatility;
    float volatility_skew;
    
    // Time features
    float hour_of_day;
    float day_of_week;
    float is_market_open;
};

// Market data processor class
class MarketDataProcessor {
private:
    // Data storage
    std::deque<std::vector<float>> price_history_;
    std::deque<OrderBookSnapshot> order_book_history_;
    std::deque<MarketTrade> trade_history_;
    std::deque<NewsItem> news_history_;
    std::deque<SentimentData> sentiment_history_;
    
    // Feature calculation
    std::unordered_map<std::string, std::function<float(const MarketDataProcessor&)>> feature_calculators_;
    
    // Configuration
    int price_history_length_;
    int order_book_history_length_;
    int trade_history_length_;
    int news_history_length_;
    int sentiment_history_length_;
    
    // Feature calculation methods
    float calculateSMA(int period) const;
    float calculateEMA(int period) const;
    float calculateRSI(int period) const;
    float calculateMACD() const;
    float calculateBollingerBand(bool upper) const;
    float calculateATR(int period) const;
    float calculateOrderBookPressure() const;
    float calculateBidAskImbalance() const;
    float calculateSentimentScore() const;
    float calculateVolatility(int period) const;
    
public:
    MarketDataProcessor(int price_history = 100, 
                       int order_book_history = 50,
                       int trade_history = 1000,
                       int news_history = 100,
                       int sentiment_history = 100);
    
    // Data ingestion
    void addPriceData(const std::vector<float>& ohlcv);
    void addOrderBookSnapshot(const OrderBookSnapshot& snapshot);
    void addTrade(const MarketTrade& trade);
    void addNewsItem(const NewsItem& news);
    void addSentimentData(const SentimentData& sentiment);
    
    // Feature extraction
    MarketFeatures extractFeatures() const;
    std::vector<float> getNormalizedFeatureVector() const;
    
    // Data loading
    bool loadPriceDataFromCSV(const std::string& filename);
    bool loadOrderBookDataFromCSV(const std::string& filename);
    bool loadTradeDataFromCSV(const std::string& filename);
    bool loadNewsDataFromCSV(const std::string& filename);
    bool loadSentimentDataFromCSV(const std::string& filename);
    
    // Real-time data connection
    bool connectToDataFeed(const std::string& feed_url, const std::string& api_key);
    void startDataStream();
    void stopDataStream();
    
    // Utility
    void clearAllData();
    void setFeatureNormalizationParams(const std::vector<std::pair<float, float>>& params);
};

#endif // MARKET_DATA_PROCESSOR_H
```

#### 2.2 Implement Order Book Processing

```cpp
// OrderBookProcessor.h
#ifndef ORDER_BOOK_PROCESSOR_H
#define ORDER_BOOK_PROCESSOR_H

#include <vector>
#include <map>
#include <string>

// Order book level
struct OrderBookLevel {
    float price;
    float volume;
    int order_count;
};

// Order book snapshot
class OrderBookSnapshot {
private:
    std::vector<OrderBookLevel> bids_;
    std::vector<OrderBookLevel> asks_;
    std::string symbol_;
    long long timestamp_;
    
public:
    OrderBookSnapshot(const std::string& symbol, long long timestamp);
    
    // Accessors
    const std::vector<OrderBookLevel>& getBids() const { return bids_; }
    const std::vector<OrderBookLevel>& getAsks() const { return asks_; }
    const std::string& getSymbol() const { return symbol_; }
    long long getTimestamp() const { return timestamp_; }
    
    // Modifiers
    void addBid(float price, float volume, int order_count);
    void addAsk(float price, float volume, int order_count);
    void clear();
    
    // Analysis
    float getBidAskSpread() const;
    float getMidPrice() const;
    float getWeightedMidPrice() const;
    float getBidAskImbalance() const;
    float getOrderBookPressure() const;
    float getVolumeAtLevel(bool is_bid, int level) const;
    float getPriceAtLevel(bool is_bid, int level) const;
};

// Full order book with updates
class OrderBook {
private:
    std::map<float, OrderBookLevel, std::greater<float>> bids_;  // Sorted high to low
    std::map<float, OrderBookLevel> asks_;  // Sorted low to high
    std::string symbol_;
    long long last_update_time_;
    
    // Event callbacks
    std::function<void(const OrderBookSnapshot&)> on_update_;
    
public:
    OrderBook(const std::string& symbol);
    
    // Order book
