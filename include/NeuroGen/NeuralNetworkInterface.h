#ifndef NEUROGEN_NEURALNETWORKINTERFACE_H
#define NEUROGEN_NEURALNETWORKINTERFACE_H

#include <string>
#include <vector>
#include <map>
#include <utility> // For std::pair

// Forward declare or include actual BNN API headers if needed by public interface
// For now, we assume BNN interaction is encapsulated in the .cpp file

class NeuralNetworkInterface {
public:
    NeuralNetworkInterface();
    ~NeuralNetworkInterface();

    /**
     * @brief Initializes the connection or interface to the BNN.
     * @return True if initialization is successful, false otherwise.
     */
    bool initialize(); // Changed to void in cpp, but bool makes more sense for success/failure

    /**
     * @brief Sends features to the BNN and retrieves a price prediction.
     * @param features A map of feature names to their values.
     * @return Vector containing the BNN's prediction signal and confidence.
     */
    std::vector<double> getPrediction(const std::map<std::string, double>& features);

    /**
     * @brief Check if the neural network interface is initialized.
     * @return True if initialized, false otherwise.
     */
    bool isInitialized() const;

    /**
     * @brief Sends a reward signal to the BNN for learning/adaptation.
     * @param reward The calculated reward value based on the outcome of a trading decision.
     */
    void sendRewardSignal(double reward);

    /**
     * @brief Saves the state of the BNN.
     * @param filename The base filename to save the BNN state to.
     * @return True if saving is successful, false otherwise.
     */
    bool saveState(const std::string& filename);

    /**
     * @brief Loads the state of the BNN.
     * @param filename The base filename to load the BNN state from.
     * @return True if loading is successful, false otherwise.
     */
    bool loadState(const std::string& filename);

private:
    // Helper method to convert feature map to a flat vector in the correct order
    std::vector<float> _convertFeaturesToInput(const std::map<std::string, double>& features);

    // Helper method to normalize features before sending to the BNN
    void _normalizeFeatures(std::vector<float>& input_vector);
    
    bool is_initialized_;
    float last_reward_; // Stores the last reward signal
    std::vector<float> last_prediction_; // Stores the last prediction from the BNN

    // Order of features expected by the BNN
    std::vector<std::string> feature_order_;
    // Min/max ranges for feature normalization
    std::map<std::string, std::pair<double, double>> feature_ranges_;
};

#endif // NEUROGEN_NEURALNETWORKINTERFACE_H
