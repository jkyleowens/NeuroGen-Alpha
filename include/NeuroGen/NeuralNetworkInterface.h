#ifndef NEUROGEN_NEURALNETWORKINTERFACE_H
#define NEUROGEN_NEURALNETWORKINTERFACE_H

#include <string>
#include <vector>
#include <map>
#include <utility> // For std::pair

class NeuralNetworkInterface {
public:
    struct Config {
        std::string model_path;
        std::vector<std::string> feature_order;
    };

    NeuralNetworkInterface(const Config& config);
    ~NeuralNetworkInterface();

    bool initialize(); 

    std::vector<double> getPrediction(const std::map<std::string, double>& features);

    bool isInitialized() const;

    void sendRewardSignal(double reward);

    /**
     * @brief Saves the state of the BNN.
     * @param filename The base filename to save the BNN state to.
     * @return True if saving is successful, false otherwise.
     */
    // **FINAL FIX**: This 'const' is essential and must match the .cpp file.
    bool saveState(const std::string& filename) const; 

    /**
     * @brief Loads the state of the BNN.
     * @param filename The base filename to load the BNN state from.
     * @return True if loading is successful, false otherwise.
     */
    bool loadState(const std::string& filename);

private:
    std::vector<float> _convertFeaturesToInput(const std::map<std::string, double>& features);
    void _normalizeFeatures(std::vector<float>& input_vector);
    
    Config config_;
    bool is_initialized_;
    float last_reward_;
    std::vector<float> last_prediction_;

    std::map<std::string, std::pair<double, double>> feature_ranges_;
};

#endif // NEUROGEN_NEURALNETWORKINTERFACE_H
