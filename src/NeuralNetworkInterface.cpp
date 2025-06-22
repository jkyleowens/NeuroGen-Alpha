#include <NeuroGen/NeuralNetworkInterface.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

// =================================================================================
// Constructor & Destructor
// =================================================================================

NeuralNetworkInterface::NeuralNetworkInterface(const Config& config)
    : config_(config),
      is_initialized_(false),
      last_reward_(0.0f) {
    // The feature order is copied from the config.
    // Initialization of the actual BNN will happen in initialize().
    std::cout << "[NeuralNetworkInterface] Constructor called." << std::endl;
}

NeuralNetworkInterface::~NeuralNetworkInterface() {
    // Perform any necessary cleanup of the BNN resources.
    std::cout << "[NeuralNetworkInterface] Destructor called." << std::endl;
}

// =================================================================================
// Public Methods
// =================================================================================

bool NeuralNetworkInterface::initialize() {
    // Placeholder: In a real scenario, this would load the model from config_.model_path
    // and prepare the BNN library for execution.
    std::cout << "[NeuralNetworkInterface] Initializing..." << std::endl;
    if (config_.model_path.empty()) {
        std::cerr << "Warning: NN model path is empty. Initializing in a mock state." << std::endl;
    }
    is_initialized_ = true;
    std::cout << "[NeuralNetworkInterface] Initialized successfully." << std::endl;
    return is_initialized_;
}

std::vector<double> NeuralNetworkInterface::getPrediction(const std::map<std::string, double>& features) {
    if (!is_initialized_) {
        return {}; // Return empty vector if not initialized
    }
    // Placeholder: This would normally convert features, run them through the BNN,
    // and return the output. For now, it returns a mock prediction.
    double mock_signal = 0.0;
    if (features.count("price_change_pct")) {
        // Simple mock logic: if price went up, predict a positive signal.
        mock_signal = features.at("price_change_pct") * 10.0;
    }
    double mock_confidence = std::min(1.0, std::abs(mock_signal));
    
    return {mock_signal, mock_confidence};
}

bool NeuralNetworkInterface::isInitialized() const {
    return is_initialized_;
}

void NeuralNetworkInterface::sendRewardSignal(double reward) {
    if (!is_initialized_) return;
    last_reward_ = static_cast<float>(reward);
    // Placeholder: This would pass the reward to the BNN's learning algorithm.
    std::cout << "[NeuralNetworkInterface] Received reward signal: " << reward << std::endl;
}


/**
 * @brief Saves the current state of the neural network to a file.
 *
 * **CRITICAL FIX**: The 'const' keyword is added here to match the declaration
 * in the header file. This resolves the 'discards qualifiers' error by
 * promising the compiler that this function will not modify the object's state.
 */
bool NeuralNetworkInterface::saveState(const std::string& filename) const {
    if (!is_initialized_) {
        std::cerr << "Error: Cannot save state of uninitialized NeuralNetworkInterface." << std::endl;
        return false;
    }

    std::string full_filename = filename + "_nn_state.json";
    std::ofstream state_file(full_filename);

    if (!state_file.is_open()) {
        std::cerr << "Error: Could not open file for writing NN state: " << full_filename << std::endl;
        return false;
    }

    nlohmann::json nn_state;
    nn_state["model_path"] = config_.model_path;
    nn_state["feature_order"] = config_.feature_order;
    nn_state["last_reward"] = last_reward_;

    // In a real implementation, you would serialize the BNN model weights here.
    
    state_file << nn_state.dump(4); // Save the JSON state with pretty printing
    std::cout << "[NeuralNetworkInterface] State saved successfully to " << full_filename << std::endl;
    return true;
}

bool NeuralNetworkInterface::loadState(const std::string& filename) {
    std::string full_filename = filename + "_nn_state.json";
    std::ifstream state_file(full_filename);
    if (!state_file.is_open()) {
        std::cerr << "Error: Could not open NN state file for reading: " << full_filename << std::endl;
        return false;
    }
    
    // In a real implementation, you would load the BNN model weights here.

    is_initialized_ = true; // Mark as initialized after successful load.
    std::cout << "[NeuralNetworkInterface] State loaded successfully from " << full_filename << std::endl;
    return true;
}

// =================================================================================
// Private Helper Methods
// =================================================================================

std::vector<float> NeuralNetworkInterface::_convertFeaturesToInput(const std::map<std::string, double>& features) {
    // Implementation would convert map to ordered vector based on config_.feature_order
    return {};
}

void NeuralNetworkInterface::_normalizeFeatures(std::vector<float>& input_vector) {
    // Implementation would normalize features based on stored min/max ranges.
}