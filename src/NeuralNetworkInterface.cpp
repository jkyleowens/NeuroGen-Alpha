#include "NeuralNetworkInterface.h" // Correct include path
#include <NeuroGen/cuda/NetworkCUDA_Interface.h> // BNN API
#include <NeuroGen/NetworkConfig.h> // For NetworkConfig if needed, though not directly used here
#include <iostream>
#include <algorithm> // For std::transform
#include <cmath>     // For std::isnan, std::isinf
#include <fstream>   // For saveState/loadState if implemented with file I/O (BNN API handles this)
#include <sstream>   // For string manipulation if needed

// Constructor
NeuralNetworkInterface::NeuralNetworkInterface()
    : is_initialized_(false), last_reward_(0.0f) {
    
    // Define the order of features to ensure consistent input vector for the BNN
    // This order must match the BNN\'s expected input layer configuration.
    // The BNN (NetworkCUDA_Interface) expects a flat vector of floats.
    // The size of this vector should match `NetworkConfig::input_size`.
    // We will use the feature_order_ to map std::map<string, double> to std::vector<float>.
    feature_order_ = {
        // Price-based features
        "price", "volume", 
        "price_change", "price_change_pct", // Raw price changes
        // SMA features
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        // EMA features
        "ema_12", "ema_26",
        // MACD features
        "macd", "macd_signal", "macd_histogram",
        // RSI
        "rsi_14",
        // Bollinger Bands
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
        // ATR
        "atr_14",
        // OBV
        "obv",
        // Stochastic Oscillator
        "stoch_14", // Typically %K
        // Momentum
        "momentum_5", "momentum_10", "momentum_20",
        // Crossover signals (binary or scaled)
        "sma_5_10_cross", "sma_10_20_cross", "sma_50_200_cross",
        // Volatility
        "volatility_20" 
        // Add more features as defined by NetworkConfig::input_size
        // Ensure this list matches the expected input features of the BNN.
        // The current list has 30 features. If NetworkConfig::input_size is different,
        // this list needs to be adjusted.
    };
    
    // Initialize feature ranges with plausible default values (min, max)
    // These are crucial for normalization.
    // Values should be based on expected ranges of these technical indicators.
    for (const auto& feature_name : feature_order_) {
        // Default to a wide range, adjust specific ones below
        feature_ranges_[feature_name] = {-1e6, 1e6}; 
    }

    // Specific ranges for common indicators:
    feature_ranges_["price"] = {0.0, 1e7}; // e.g., BTC price
    feature_ranges_["volume"] = {0.0, 1e9}; // Large volume for major pairs

    feature_ranges_["price_change"] = {-1000.0, 1000.0}; // Absolute price change
    feature_ranges_["price_change_pct"] = {-0.2, 0.2}; // -20% to +20% change

    feature_ranges_["sma_5"] = {0.0, 1e7};
    feature_ranges_["sma_10"] = {0.0, 1e7};
    feature_ranges_["sma_20"] = {0.0, 1e7};
    feature_ranges_["sma_50"] = {0.0, 1e7};
    feature_ranges_["sma_200"] = {0.0, 1e7};

    feature_ranges_["ema_12"] = {0.0, 1e7};
    feature_ranges_["ema_26"] = {0.0, 1e7};

    feature_ranges_["macd"] = {-1000.0, 1000.0}; // Can be negative
    feature_ranges_["macd_signal"] = {-1000.0, 1000.0};
    feature_ranges_["macd_histogram"] = {-500.0, 500.0};

    feature_ranges_["rsi_14"] = {0.0, 100.0}; // Standard RSI range
    
    feature_ranges_["bb_upper"] = {0.0, 1e7};
    feature_ranges_["bb_middle"] = {0.0, 1e7}; // Same as an SMA
    feature_ranges_["bb_lower"] = {0.0, 1e7};
    feature_ranges_["bb_width"] = {0.0, 0.5}; // BB width as a percentage of middle band
    feature_ranges_["bb_position"] = {0.0, 1.0}; // Price position relative to bands

    feature_ranges_["atr_14"] = {0.0, 1000.0}; // Average True Range
    feature_ranges_["obv"] = {-1e10, 1e10}; // On-Balance Volume can be large and vary widely

    feature_ranges_["stoch_14"] = {0.0, 100.0}; // Stochastic %K

    feature_ranges_["momentum_5"] = {-1000.0, 1000.0}; 
    feature_ranges_["momentum_10"] = {-1000.0, 1000.0};
    feature_ranges_["momentum_20"] = {-1000.0, 1000.0};

    // Crossover signals might be binary (e.g., -1, 0, 1) or scaled.
    // Assuming scaled for now, normalized later.
    feature_ranges_["sma_5_10_cross"] = {-1.0, 1.0}; 
    feature_ranges_["sma_10_20_cross"] = {-1.0, 1.0};
    feature_ranges_["sma_50_200_cross"] = {-1.0, 1.0};
    
    feature_ranges_["volatility_20"] = {0.0, 0.2}; // e.g., 20-day price volatility (std dev)
}

NeuralNetworkInterface::~NeuralNetworkInterface() {
    if (is_initialized_) {
        // The BNN API should provide a cleanup function.
        // Assuming `cleanupNetwork()` is the correct function from NetworkCUDA_Interface.h
        std::cout << "[NeuralNetworkInterface] Cleaning up BNN..." << std::endl;
        cleanupNetwork(); 
    }
}

bool NeuralNetworkInterface::initialize() {
    if (is_initialized_) {
        std::cout << "[NeuralNetworkInterface] Already initialized." << std::endl;
        return true;
    }
    
    std::cout << "[NeuralNetworkInterface] Initializing BNN..." << std::endl;
    
    // Call the BNN\'s initialization function.
    // `initializeNetwork()` is from `NetworkCUDA_Interface.h`.
    // It\'s a void function, so we assume success if it doesn\'t throw an exception.
    // A robust BNN API might return a status or allow querying config.
    try {
        // Optional: Configure the network if NetworkConfig is used by the BNN
        // NetworkConfig config; // Create a config object
        // // ... set config parameters ...
        // NetworkConfig currentConfig = getNetworkConfig(); // Get default/current
        // currentConfig.input_size = feature_order_.size(); 
        // // currentConfig.output_size = ...; // Define based on desired output (e.g., buy/sell/hold signal, price prediction)
        // setNetworkConfig(currentConfig); // Apply custom config if supported

        initializeNetwork(); // This is the primary initialization call.
        is_initialized_ = true;
        std::cout << "[NeuralNetworkInterface] BNN initialized successfully." << std::endl;
        
        // Verify input size matches BNN configuration if possible
        NetworkConfig active_config = getNetworkConfig(); // Assumes BNN provides this
        if (active_config.input_size != feature_order_.size()) {
            std::cerr << "[NeuralNetworkInterface] Warning: Feature order size (" << feature_order_.size() 
                      << ") does not match BNN input_size (" << active_config.input_size 
                      << "). This may lead to errors." << std::endl;
            // Potentially, this should be a fatal error, or the BNN should be reconfigured.
        }

    } catch (const std::exception& e) {
        std::cerr << "[NeuralNetworkInterface] BNN initialization failed: " << e.what() << std::endl;
        is_initialized_ = false;
        return false;
    } catch (...) {
        std::cerr << "[NeuralNetworkInterface] BNN initialization failed due to an unknown error." << std::endl;
        is_initialized_ = false;
        return false;
    }
    
    return is_initialized_;
}

// Helper method to convert feature map to a flat vector in the correct order
std::vector<float> NeuralNetworkInterface::_convertFeaturesToInput(const std::map<std::string, double>& features) {
    std::vector<float> input_vector;
    input_vector.reserve(feature_order_.size());

    for (const std::string& feature_name : feature_order_) {
        auto it = features.find(feature_name);
        if (it != features.end()) {
            // Handle potential NaN or Inf values from technical analysis
            double value = it->second;
            if (std::isnan(value) || std::isinf(value)) {
                // Replace NaN/Inf with a neutral value (e.g., 0 or midpoint of its range)
                // Or, use the previous valid value if available. For simplicity, using 0.
                input_vector.push_back(0.0f); 
                // std::cerr << "[NeuralNetworkInterface] Warning: NaN/Inf found for feature \'" << feature_name << "\'. Using 0.0f." << std::endl;
            } else {
                input_vector.push_back(static_cast<float>(value));
            }
        } else {
            // If a feature is missing, add a default value (e.g., 0.0)
            // This should ideally not happen if TechnicalAnalysis provides all features.
            input_vector.push_back(0.0f); 
            std::cerr << "[NeuralNetworkInterface] Warning: Feature \'" << feature_name << "\' not found in input map. Using 0.0f." << std::endl;
        }
    }
    return input_vector;
}

// Helper method to normalize features before sending to the BNN
// Normalizes to a common range, e.g., [0, 1] or [-1, 1], based on BNN requirements.
// Assuming BNN prefers inputs in [0, 1] range for simplicity.
void NeuralNetworkInterface::_normalizeFeatures(std::vector<float>& input_vector) {
    if (input_vector.size() != feature_order_.size()) {
        std::cerr << "[NeuralNetworkInterface] Error: Input vector size mismatch during normalization." << std::endl;
        return;
    }

    for (size_t i = 0; i < input_vector.size(); ++i) {
        const std::string& feature_name = feature_order_[i];
        auto range_it = feature_ranges_.find(feature_name);

        if (range_it != feature_ranges_.end()) {
            double min_val = range_it->second.first;
            double max_val = range_it->second.second;
            
            if (max_val - min_val == 0) { // Avoid division by zero
                 input_vector[i] = 0.5f; // Or some other neutral value
            } else {
                // Normalize to [0, 1]
                float normalized_value = (input_vector[i] - static_cast<float>(min_val)) / static_cast<float>(max_val - min_val);
                // Clamp to [0, 1] to handle outliers
                input_vector[i] = std::max(0.0f, std::min(1.0f, normalized_value));
            }
        } else {
            std::cerr << "[NeuralNetworkInterface] Warning: No range defined for feature \'" << feature_name << "\'. Skipping normalization for it." << std::endl;
            // Optionally, apply a default normalization or leave as is if BNN can handle it.
        }
    }
}


std::vector<double> NeuralNetworkInterface::getPrediction(const std::map<std::string, double>& features) {
    if (!is_initialized_) {
        std::cerr << "[NeuralNetworkInterface] Error: BNN not initialized. Cannot get prediction." << std::endl;
        return {}; // Return empty vector for error
    }
    
    // 1. Convert feature map to ordered input vector
    std::vector<float> input_vector = _convertFeaturesToInput(features);

    // 2. Normalize features (if BNN requires it)
    // The BNN might do its own internal normalization or expect raw/scaled values.
    // Assuming normalization to [0,1] is beneficial or required.
    _normalizeFeatures(input_vector);
    
    // 3. Send to BNN and get output
    // `forwardCUDA` is from `NetworkCUDA_Interface.h`.
    // It takes the input vector and the last reward signal.
    // std::cout << "[NeuralNetworkInterface] Sending features to BNN. Input vector size: " << input_vector.size() << std::endl;
    // for(size_t i=0; i < input_vector.size(); ++i) {
    //     std::cout << "  Input[" << i << "] (" << feature_order_[i] << "): " << input_vector[i] << std::endl;
    // }
    // std::cout << "[NeuralNetworkInterface] Last reward signal: " << last_reward_ << std::endl;

    std::vector<float> bnn_output;
    try {
        bnn_output = forwardCUDA(input_vector, last_reward_);
    } catch (const std::exception& e) {
        std::cerr << "[NeuralNetworkInterface] Error during BNN forward pass: " << e.what() << std::endl;
        return {}; // Return empty vector for error
    } catch (...) {
        std::cerr << "[NeuralNetworkInterface] Unknown error during BNN forward pass." << std::endl;
        return {}; // Return empty vector for error
    }
    
    last_prediction_ = bnn_output; // Store the raw BNN output

    // 4. Interpret BNN output
    // The BNN output format needs to be defined. Examples:
    //  - A single value representing predicted price/price change.
    //  - Multiple values (e.g., probabilities for buy/sell/hold).
    //  - Activation levels of output neurons.
    //  Let\'s assume `NetworkConfig::output_size` defines this.
    //  For now, assume the first output value is the primary prediction signal.
    //  This signal might be e.g. a value in [-1, 1] where -1 is strong sell, 1 is strong buy.

    if (bnn_output.empty()) {
        std::cerr << "[NeuralNetworkInterface] BNN returned empty output." << std::endl;
        return {}; // Return empty vector for error
    }

    // Example: Interpret the first output neuron's activation as the decision signal.
    // This signal could be a raw value, or it might need scaling/interpretation.
    // Let's assume it's a value that can be directly used or scaled.
    // For a trading agent, this might represent confidence in price increase (positive) or decrease (negative).
    double prediction_signal = static_cast<double>(bnn_output[0]);
    
    // Return both signal and confidence (if available)
    std::vector<double> result;
    result.push_back(prediction_signal);
    
    // If BNN provides multiple outputs, use second as confidence, otherwise derive it
    if (bnn_output.size() > 1) {
        double confidence = static_cast<double>(bnn_output[1]);
        result.push_back(confidence);
    } else {
        // Derive confidence from signal strength
        double confidence = std::abs(prediction_signal);
        result.push_back(confidence);
    }

    // The `AutonomousTradingAgent` will use this signal to make a decision (buy/sell/hold).
    // The interpretation of this value (e.g. mapping to buy/sell thresholds) happens in the Agent.
    // Here, we just return the BNN's direct (or minimally processed) output.
    
    // For debugging:
    // std::cout << "[NeuralNetworkInterface] BNN Output (raw): ";
    // for(float val : bnn_output) { std::cout << val << " "; }
    // std::cout << std::endl;
    // std::cout << "[NeuralNetworkInterface] Prediction signal (output[0]): " << prediction_signal << std::endl;

    return result; 
}

void NeuralNetworkInterface::sendRewardSignal(double reward) {
    if (!is_initialized_) {
        std::cerr << "[NeuralNetworkInterface] Error: BNN not initialized. Cannot send reward signal." << std::endl;
        return;
    }
    
    last_reward_ = static_cast<float>(reward);
    // std::cout << "[NeuralNetworkInterface] Received reward signal: " << reward << ". Stored as: " << last_reward_ << std::endl;

    // The BNN API (`NetworkCUDA_Interface.h`) has `updateSynapticWeightsCUDA(float reward_signal)`.
    // This seems like the correct place to apply the reward for learning.
    // The `forwardCUDA` function also takes `reward_signal`. This suggests the BNN might use
    // the reward signal during the forward pass (e.g., for neuromodulation or eligibility traces)
    // AND/OR for a separate weight update step.
    //
    // If `updateSynapticWeightsCUDA` is the primary learning trigger:
    try {
        // std::cout << "[NeuralNetworkInterface] Calling updateSynapticWeightsCUDA with reward: " << last_reward_ << std::endl;
        updateSynapticWeightsCUDA(last_reward_);
    } catch (const std::exception& e) {
        std::cerr << "[NeuralNetworkInterface] Error during BNN weight update: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[NeuralNetworkInterface] Unknown error during BNN weight update." << std::endl;
    }

    // If the BNN also uses reward in forward pass, `last_reward_` will be passed in the next `getPrediction` call.
    // This design (reward passed to forward and to a separate update function) is common in some RL models.
}

bool NeuralNetworkInterface::saveState(const std::string& filename) {
    if (!is_initialized_) {
        std::cerr << "[NeuralNetworkInterface] Error: BNN not initialized. Cannot save state." << std::endl;
        return false;
    }
    
    std::cout << "[NeuralNetworkInterface] Saving BNN state to file: " << filename << std::endl;
    try {
        // `saveNetworkState` is from `NetworkCUDA_Interface.h`.
        saveNetworkState(filename);
        std::cout << "[NeuralNetworkInterface] BNN state saved successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[NeuralNetworkInterface] Failed to save BNN state: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[NeuralNetworkInterface] Failed to save BNN state due to an unknown error." << std::endl;
        return false;
    }
}

bool NeuralNetworkInterface::loadState(const std::string& filename) {
    // BNN must be initialized before loading state, or loadState should handle initialization.
    // Assuming `initializeNetwork()` should be called first.
    if (!is_initialized_) {
        std::cout << "[NeuralNetworkInterface] BNN not initialized. Initializing before loading state..." << std::endl;
        if (!initialize()) { // Try to initialize
             std::cerr << "[NeuralNetworkInterface] Initialization failed. Cannot load state." << std::endl;
             return false;
        }
    }
    
    std::cout << "[NeuralNetworkInterface] Loading BNN state from file: " << filename << std::endl;
    try {
        // `loadNetworkState` is from `NetworkCUDA_Interface.h`.
        loadNetworkState(filename);
        is_initialized_ = true; // Ensure state is marked as initialized after loading
        std::cout << "[NeuralNetworkInterface] BNN state loaded successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[NeuralNetworkInterface] Failed to load BNN state: " << e.what() << std::endl;
        // Depending on BNN behavior, a failed load might leave it in an unusable state.
        // is_initialized_ = false; // Consider this if a failed load corrupts the BNN instance.
        return false;
    } catch (...) {
        std::cerr << "[NeuralNetworkInterface] Failed to load BNN state due to an unknown error." << std::endl;
        return false;
    }
}

bool NeuralNetworkInterface::isInitialized() const {
    return is_initialized_;
}
