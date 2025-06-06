/**
 * @file Network.h
 * @brief Biologically Inspired Dynamic Neural Network with Neurogenesis, Synaptogenesis, and Pruning
 * 
 * This header defines a comprehensive neural network system that simulates biological
 * processes including activity-driven connectivity, dynamic synapse formation,
 * neurogenesis, and synaptic pruning. The network supports 3D spatial organization
 * and implements realistic plasticity mechanisms.
 * 
 * @author Neural Dynamics Lab
 * @version 2.0
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <functional>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>

#if defined(USE_CUDA) && USE_CUDA
#include "NeuroGen/cuda/NetworkCUDA.cuh"
#endif

#include "Neuron.h"

// Optional JSON support - disabled for now to avoid dependency issues
// #define USE_JSON_CONFIG
#ifdef USE_JSON_CONFIG
#include <json/json.h>
#endif

// Forward declarations
class Neuron;
class Compartment;
class IonChannel;
class SynapticReceptor;

/**
 * @brief 3D position structure for spatial neuron organization
 */
struct Position3D {
    double x, y, z;
    
    Position3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
    
    double distanceTo(const Position3D& other) const {
        double dx = x - other.x, dy = y - other.y, dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    Position3D operator+(const Position3D& other) const {
        return Position3D(x + other.x, y + other.y, z + other.z);
    }
    
    Position3D operator*(double scalar) const {
        return Position3D(x * scalar, y * scalar, z * scalar);
    }
};

/**
 * @brief Spike event with timing and neuron identification
 */
struct SpikeEvent {
    size_t neuron_id;
    double timestamp;
    double amplitude;
    
    SpikeEvent(size_t id, double time, double amp = 1.0) 
        : neuron_id(id), timestamp(time), amplitude(amp) {}
    
    bool operator<(const SpikeEvent& other) const {
        return timestamp > other.timestamp; // For min-heap ordering
    }
};

/**
 * @brief Synaptic connection between neurons with plasticity tracking
 */
struct Synapse {
    size_t pre_neuron_id;
    size_t post_neuron_id;
    std::string post_compartment;
    size_t receptor_index;
    
    double weight;
    double base_weight;
    double axonal_delay;        // ms
    
    // Plasticity tracking
    double last_pre_spike;
    double last_post_spike;
    double eligibility_trace;
    double activity_metric;     // Running average of usage
    
    // Structural plasticity
    double formation_time;
    double last_potentiation;
    double strength_history[10]; // Sliding window for pruning decisions
    size_t history_index;
    
    Synapse(size_t pre_id, size_t post_id, const std::string& compartment, 
            size_t receptor_idx, double w = 0.1, double delay = 1.0)
        : pre_neuron_id(pre_id), post_neuron_id(post_id), 
          post_compartment(compartment), receptor_index(receptor_idx),
          weight(w), base_weight(w), axonal_delay(delay),
          last_pre_spike(-1000.0), last_post_spike(-1000.0),
          eligibility_trace(0.0), activity_metric(0.0),
          formation_time(0.0), last_potentiation(0.0), history_index(0) {
        std::fill(std::begin(strength_history), std::end(strength_history), w);
    }
};

/**
 * @brief Neuromodulator system for network-wide plasticity modulation
 */
struct NeuromodulatorSystem {
    double dopamine;        // Reward/reinforcement
    double acetylcholine;   // Attention/learning
    double norepinephrine;  // Arousal/stress
    double serotonin;       // Mood/inhibition modulation
    
    // Decay time constants (ms)
    double da_tau = 100.0;
    double ach_tau = 50.0;
    double ne_tau = 200.0;
    double ser_tau = 1000.0;
    
    NeuromodulatorSystem() : dopamine(0.0), acetylcholine(0.0), 
                           norepinephrine(0.0), serotonin(0.0) {}
    
    void decay(double dt) {
        dopamine *= std::exp(-dt / da_tau);
        acetylcholine *= std::exp(-dt / ach_tau);
        norepinephrine *= std::exp(-dt / ne_tau);
        serotonin *= std::exp(-dt / ser_tau);
    }
    
    void releaseDopamine(double amount) { dopamine += amount; }
    void releaseAcetylcholine(double amount) { acetylcholine += amount; }
    void releaseNorepinephrine(double amount) { norepinephrine += amount; }
    void releaseSerotonin(double amount) { serotonin += amount; }
};

/**
 * @brief Network configuration parameters
 */
struct NetworkConfig {
    // Simulation parameters
    double dt = 0.01;                    // Integration time step (ms)
    double axonal_speed = 1.0;           // m/s (affects delays)
    
    // Spatial organization
    double network_width = 1000.0;       // μm
    double network_height = 1000.0;      // μm
    double network_depth = 100.0;        // μm
    
    // Connectivity parameters
    double max_connection_distance = 200.0; // μm
    double connection_probability_base = 0.01;
    double distance_decay_constant = 50.0;   // μm
    double spike_correlation_window = 20.0;  // ms
    double correlation_threshold = 0.3;
    
    // Neurogenesis parameters
    bool enable_neurogenesis = true;
    double neurogenesis_rate = 0.001;        // neurons/ms base rate
    double activity_threshold_low = 0.1;     // For underactivation
    double activity_threshold_high = 10.0;   // For hyperactivation
    size_t max_neurons = 1000;
    
    // Pruning parameters
    bool enable_pruning = true;
    double synapse_pruning_threshold = 0.05;
    double neuron_pruning_threshold = 0.01;
    double pruning_check_interval = 100.0;   // ms
    double synapse_activity_window = 1000.0; // ms
    
    // Plasticity parameters
    bool enable_stdp = true;
    double stdp_learning_rate = 0.01;
    double stdp_tau_pre = 20.0;              // ms
    double stdp_tau_post = 20.0;             // ms
    double eligibility_decay = 50.0;         // ms
    double min_synaptic_weight = 0.001;      // Minimum synaptic weight
    double max_synaptic_weight = 2.0;        // Maximum synaptic weight
    
    // Neuromodulation
    bool enable_neuromodulation = true;
    double modulation_strength = 0.1;

    std::string toString() const {
        // Basic implementation, can be expanded
        return "NetworkConfig{dt=" + std::to_string(dt) +
               ", max_neurons=" + std::to_string(max_neurons) + "}";
    }
};

/**
 * @brief Neuron factory for creating different neuron types
 */
class NeuronFactory {
public:
    enum NeuronType {
        PYRAMIDAL_CORTICAL,
        INTERNEURON_BASKET,
        INTERNEURON_CHANDELIER,
        PURKINJE_CEREBELLAR,
        GRANULE_CEREBELLAR,
        SENSORY, // Added
        MOTOR,   // Added
        CUSTOM
    };
    
    /**
     * @brief Create a neuron of specified type at given position
     */
    static std::shared_ptr<Neuron> createNeuron(NeuronType type, 
                                               const std::string& id, 
                                               const Position3D& position);
    
    /**
     * @brief Register a custom neuron creation function
     */
    static void registerCustomType(const std::string& type_name,
                                 std::function<std::shared_ptr<Neuron>(const std::string&, const Position3D&)> creator);

private:
    static std::unordered_map<std::string, std::function<std::shared_ptr<Neuron>(const std::string&, const Position3D&)>> custom_creators_;
};

/**
 * @brief Main Network class managing dynamic neural populations
 */
class Network {
public:
    enum LayerType {
        INPUT,
        HIDDEN,
        OUTPUT
    };

    /**
     * @brief Construct network with configuration
     */
    explicit Network(const NetworkConfig& config = NetworkConfig());
    
    /**
     * @brief Destructor
     */
    ~Network();
    
    // === Core Network Management ===
    
    /**
     * @brief Add a neuron to the network at specified position
     */
    size_t addNeuron(std::shared_ptr<Neuron> neuron, const Position3D& position); // Returns neuron ID
    
    /**
     * @brief Remove a neuron from the network
     */
    bool removeNeuron(size_t neuron_id);
    
    /**
     * @brief Get neuron by ID
     */
    std::shared_ptr<Neuron> getNeuron(size_t neuron_id) const;
    
    /**
     * @brief Get neuron position
     */
    Position3D getNeuronPosition(size_t neuron_id) const;
    
    /**
     * @brief Set neuron position (for migration/growth)
     */
    void setNeuronPosition(size_t neuron_id, const Position3D& position);
    
    // === Connectivity Management ===
    
    /**
     * @brief Create synapse between neurons
     */
    bool createSynapse(size_t pre_neuron_id, size_t post_neuron_id,
                      const std::string& post_compartment, size_t receptor_index,
                      double weight = 0.1, double delay = 1.0);
    
    /**
     * @brief Remove synapse
     */
    bool removeSynapse(size_t pre_neuron_id, size_t post_neuron_id,
                      const std::string& post_compartment, size_t receptor_index);
    
    /**
     * @brief Get all synapses from a neuron
     */
    std::vector<Synapse*> getOutgoingSynapses(size_t neuron_id);
    
    /**
     * @brief Get all synapses to a neuron
     */
    std::vector<Synapse*> getIncomingSynapses(size_t neuron_id);
    
    // === Simulation Control ===
    
    /**
     * @brief Advance network one time step
     */
    void step(double dt);
    
    /**
     * @brief Run simulation for specified duration
     */
    void run(double duration, double dt_sim);
    
    /**
     * @brief Reset network state
     */
    void reset();
    
    // === Activity-Driven Processes ===
    
    /**
     * @brief Perform synaptogenesis based on activity and proximity
     */
    void performSynaptogenesis();
    
    /**
     * @brief Create new neurons based on network activity
     */
    void performNeurogenesis();
    
    /**
     * @brief Remove weak synapses and inactive neurons
     */
    void performPruning();
    
    /**
     * @brief Update synaptic weights using STDP
     */
    void updatePlasticity();
    
    // === Neuromodulation ===
    
    /**
     * @brief Release neuromodulator into network
     */
    void releaseNeuromodulator(const std::string& type, double amount);
    
    /**
     * @brief Get current neuromodulator levels
     */
    const NeuromodulatorSystem& getNeuromodulators() const { return neuromodulators_; }
    
    // === Analysis and Monitoring ===
    
    /**
     * @brief Calculate network activity statistics
     */
    struct NetworkStats {
        double mean_firing_rate;
        double network_synchrony;
        double mean_connectivity;
        double excitation_inhibition_ratio;
        size_t active_neurons;
        size_t total_synapses;
        double mean_synaptic_strength;
    };
    
    NetworkStats calculateNetworkStats(double time_window = 1000.0) const;
    
    /**
     * @brief Get regional activity (divided into spatial bins)
     */
    std::vector<double> getRegionalActivity(size_t x_bins = 10, size_t y_bins = 10, 
                                          double time_window = 100.0) const;
    
    /**
     * @brief Export network state to file
     */
    bool exportToFile(const std::string& filename) const;
    
    /**
     * @brief Import network state from file
     */
    bool importFromFile(const std::string& filename);
    
    // === Configuration ===
    
    /**
     * @brief Load configuration from JSON file
     */
    bool loadConfig(const std::string& filename);
    
    /**
     * @brief Save configuration to JSON file
     */
    bool saveConfig(const std::string& filename) const;
    
    // === Getters ===
    
    size_t getNumNeurons() const { return neurons_.size(); }
    size_t getNumSynapses() const { return synapses_.size(); }
    double getCurrentTime() const { return current_time_; }
    const NetworkConfig& getConfig() const { return config_; }
    
    // === Event Injection (for testing/stimulation) ===
    
    /**
     * @brief Inject external current into specific neuron
     */
    void injectCurrent(size_t neuron_id, double current);
    
    /**
     * @brief Stimulate neuron with spike train
     */
    void stimulateNeuron(size_t neuron_id, const std::vector<double>& spike_times);
    
    /**
     * @brief Add external synaptic input
     */
    void addExternalInput(size_t neuron_id, const std::string& compartment,
                         size_t receptor_idx, double spike_time);

    // === Layered Architecture ===
    void initializeLayeredArchitecture(size_t inputSize, size_t initialHiddenSize, size_t outputSize);
    void evaluateInputLayerSize(size_t requiredInputSize);
    void evaluateOutputLayerSize(size_t requiredOutputSize);
    void optimizeHiddenLayer();
    size_t addNeuronToLayer(LayerType layer, std::shared_ptr<Neuron> neuron, const Position3D& position);
    void connectLayers();
    // Updated declaration to include decayConstant
    void createDistanceBasedConnections(const std::vector<size_t>& source, const std::vector<size_t>& target, double probability, double maxDistance, double decayConstant);
    void updateLayeredNeurogenesisAndPruning();
    void setInputValues(const std::vector<double>& inputs);
    std::vector<double> getOutputValues() const;
    LayerType getLayerType(size_t neuron_id) const;

protected:
    // === Protected accessor methods for derived classes ===
    
    /**
     * @brief Get active neuron count
     */
    size_t getActiveNeuronCount() const { return active_neuron_ids_.size(); }
    
    /**
     * @brief Get synapse count
     */
    size_t getSynapseCount() const { return synapses_.size(); }
    
    /**
     * @brief Set current time (for derived class time management)
     */
    void setCurrentTime(double time) { current_time_ = time; }

private:
    // === Private Methods ===

    // === Layered Architecture Parameters & Data ===
    std::vector<size_t> inputLayerNeurons_;
    std::vector<size_t> hiddenLayerNeurons_;
    std::vector<size_t> outputLayerNeurons_;
    size_t maxHiddenLayerSize_ = 1048;

    // Connectivity parameters - replacing old ones
    // Inter-layer connections (Input->Hidden, Hidden->Output)
    double interLayerProbability_ = 0.1;
    double interLayerMaxDistance_ = 150.0;
    double interLayerDecayConstant_ = 150.0;

    // Recurrent connections for Hidden layer
    double hiddenRecurrentProbability_ = 0.05;
    double hiddenRecurrentMaxDistance_ = 150.0;
    double hiddenRecurrentDecayConstant_ = 100.0;

    // Recurrent connections for Input layer
    double inputRecurrentProbability_ = 0.03;
    double inputRecurrentMaxDistance_ = 75.0;
    double inputRecurrentDecayConstant_ = 50.0;

    // Recurrent connections for Output layer
    double outputRecurrentProbability_ = 0.03;
    double outputRecurrentMaxDistance_ = 75.0;
    double outputRecurrentDecayConstant_ = 50.0;

    // === Core Data Structures ===
    
    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<Position3D> neuron_positions_;
    std::unordered_set<size_t> active_neuron_ids_;
    std::unordered_set<size_t> available_neuron_ids_; // For reuse after deletion
    
    std::vector<std::unique_ptr<Synapse>> synapses_;
    std::unordered_map<size_t, std::vector<size_t>> outgoing_synapses_; // neuron_id -> synapse indices
    std::unordered_map<size_t, std::vector<size_t>> incoming_synapses_;
    
    // === Simulation State ===
    
    double current_time_;
    NetworkConfig config_;
    std::mt19937 rng_;
    
    // Event queues
    std::priority_queue<SpikeEvent> spike_queue_;
    std::vector<double> external_currents_;
    
    // Neuromodulation
    NeuromodulatorSystem neuromodulators_;
    
    // Activity tracking
    std::vector<std::vector<double>> spike_history_; // Per neuron spike times
    std::vector<double> last_activity_check_;
    double last_pruning_time_;
    double last_neurogenesis_time_;
    
    // Spatial indexing for efficient neighbor finding
    struct SpatialBin {
        std::vector<size_t> neuron_ids;
    };
    std::vector<std::vector<std::vector<SpatialBin>>> spatial_bins_;
    double bin_size_;
    
    // === Private Methods ===
    
    /**
     * @brief Initialize spatial binning system
     */
    void initializeSpatialBins();
    
    /**
     * @brief Update spatial bins after neuron movement
     */
    void updateSpatialBins();
    
    /**
     * @brief Get neurons within specified distance
     */
    std::vector<size_t> getNeuronsWithinDistance(const Position3D& center, double radius) const;
    
    /**
     * @brief Calculate connection probability based on distance and activity
     */
    double calculateConnectionProbability(size_t pre_id, size_t post_id) const;
    
    /**
     * @brief Calculate spike-time correlation between neurons
     */
    double calculateSpikeCorrelation(size_t neuron1, size_t neuron2, double time_window) const;
    
    /**
     * @brief Process delayed spike events
     */
    void processDelayedSpikes();
    
    /**
     * @brief Update eligibility traces for all synapses
     */
    void updateEligibilityTraces(double dt);
    
    /**
     * @brief Check if neuron should be pruned
     */
    bool shouldPruneNeuron(size_t neuron_id) const;
    
    /**
     * @brief Check if synapse should be pruned
     */
    bool shouldPruneSynapse(const Synapse& synapse) const;
    
    /**
     * @brief Calculate regional activity levels
     */
    std::vector<double> calculateRegionalActivityLevels() const;
    
    /**
     * @brief Get next available neuron ID
     */
    size_t getNextNeuronId();
    
    /**
     * @brief Clean up data structures after neuron removal
     */
    void cleanupAfterNeuronRemoval(size_t neuron_id);
    
    /**
     * @brief Apply neuromodulation to plasticity
     */
    double getModulatedLearningRate(double base_rate) const;
    
    /**
     * @brief Log structural changes for debugging
     */
    void logStructuralChange(const std::string& type, const std::string& details) const;

    #if defined(USE_CUDA) && USE_CUDA
    // CUDA acceleration components
    std::unique_ptr<CudaNetworkAccelerator> cuda_accelerator_;
    bool use_cuda_;
    bool cuda_data_uploaded_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point last_cuda_sync_;
    double cuda_compute_time_;
    double cpu_compute_time_;
    #endif
};

/**
 * @brief Network builder for easy network construction
 */
class NetworkBuilder {
public:
    /**
     * @brief Default constructor for NetworkBuilder.
     */
    NetworkBuilder();

    /**
     * @brief Sets the overall network configuration.
     * @param config The NetworkConfig object.
     * @return Reference to this NetworkBuilder for chaining.
     */
    NetworkBuilder& setConfig(const NetworkConfig& config);

    /**
     * @brief Adds a population of neurons to the build plan.
     * @param type The type of neurons in this population (from NeuronFactory::NeuronType).
     * @param count The number of neurons in this population.
     * @param position The central position for this population.
     * @param radius The radius around the central position within which neurons will be placed.
     * @return Reference to this NetworkBuilder for chaining.
     */
    NetworkBuilder& addNeuronPopulation(NeuronFactory::NeuronType type,
                                        unsigned long count,
                                        const Position3D& position,
                                        double radius);

    /**
     * @brief Specifies a probability for creating random connections between all neurons.
     * @param probability The probability (0.0 to 1.0) of a connection existing between any two neurons.
     * @return Reference to this NetworkBuilder for chaining.
     */
    NetworkBuilder& addRandomConnections(double probability);
    // TODO: Add addConnectionRule(const ConnectionRule& rule) if connection_rules_ are implemented.

    /**
     * @brief Constructs and returns the configured Network.
     * @return A shared_ptr to the newly created Network.
     */
    std::shared_ptr<Network> build();

private:
    /**
     * @brief Defines a population of neurons to be created.
     */
    struct NeuronPopulation {
        NeuronFactory::NeuronType type;
        unsigned long count;
        Position3D position;
        double radius;
    };

    NetworkConfig config_; ///< Configuration for the network to be built.
    std::vector<NeuronPopulation> neuronPopulations_; ///< List of neuron populations to create.
    double connectionProbability_; ///< Probability for random connections.
    // std::vector<ConnectionRule> connection_rules_; // For more specific connection strategies.

    /**
     * @brief Generates a random position within a sphere.
     * @param center The center of the sphere.
     * @param radius The radius of the sphere.
     * @return A randomized Position3D.
     */
    Position3D randomizePosition(const Position3D& center, double radius);
};


#endif // NETWORK_H