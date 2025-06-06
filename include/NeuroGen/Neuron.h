/**
 * @file Neuron.h
 * @brief Biologically Inspired Neuron Implementation Header
 * 
 * This header defines a comprehensive neuron system with multi-compartment 
 * morphology, realistic ion channels, and synaptic receptors.
 * 
 * @author Neural Dynamics Lab
 * @version 2.0
 */

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
class Compartment;
class IonChannel;
class SynapticReceptor;

/**
 * @brief Configuration structure for neuron parameters
 */
struct NeuronConfig {
    double temperature = 6.3;  // Celsius (room temperature for typical experiments)
    double dt = 0.01;         // Integration time step (ms)
    
    // Default compartment parameters
    struct CompartmentDefaults {
        double capacitance = 1.0;    // µF/cm²
        double axial_resistance = 150.0; // Ω·cm
        double length = 10.0;        // µm
        double diameter = 1.0;       // µm
    } compartment_defaults;
};

/**
 * @brief Abstract base class for all ion channels
 */
class IonChannel {
public:
    virtual ~IonChannel() = default;
    
    /**
     * @brief Calculate channel current
     * @param V Membrane voltage (mV)
     * @param dt Time step (ms)
     * @return Channel current (µA/cm²)
     */
    virtual double calculateCurrent(double V, double dt) = 0;
    
    /**
     * @brief Get channel conductance
     * @return Maximum conductance (mS/cm²)
     */
    virtual double getConductance() const = 0;
    
    /**
     * @brief Get reversal potential
     * @return Reversal potential (mV)
     */
    virtual double getReversalPotential() const = 0;
    
    /**
     * @brief Reset channel state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Get channel name for identification
     */
    virtual std::string getName() const = 0;

protected:
    /**
     * @brief Calculate rate constants alpha and beta for HH gating
     */
    struct GatingRates {
        double alpha, beta;
    };
    
    /**
     * @brief 4th-order Runge-Kutta integration for gating variables
     */
    template<typename RateFunc>
    double integrateGating(double y, double V, double dt, RateFunc rateFunc) {
        // Safety checks
        if (!std::isfinite(y) || !std::isfinite(V) || !std::isfinite(dt)) {
            return std::max(0.0, std::min(y, 1.0)); // Return clamped input if invalid
        }
        
        // Clamp input gating variable
        y = std::max(0.0, std::min(y, 1.0));
        
        auto rates1 = rateFunc(V);
        double k1 = dt * (rates1.alpha * (1.0 - y) - rates1.beta * y);
        
        double y2 = y + k1 / 2.0;
        y2 = std::max(0.0, std::min(y2, 1.0)); // Clamp intermediate values
        auto rates2 = rateFunc(V);
        double k2 = dt * (rates2.alpha * (1.0 - y2) - rates2.beta * y2);
        
        double y3 = y + k2 / 2.0;
        y3 = std::max(0.0, std::min(y3, 1.0));
        auto rates3 = rateFunc(V);
        double k3 = dt * (rates3.alpha * (1.0 - y3) - rates3.beta * y3);
        
        double y4 = y + k3;
        y4 = std::max(0.0, std::min(y4, 1.0));
        auto rates4 = rateFunc(V);
        double k4 = dt * (rates4.alpha * (1.0 - y4) - rates4.beta * y4);
        
        // Check if any k values are invalid
        if (!std::isfinite(k1) || !std::isfinite(k2) || !std::isfinite(k3) || !std::isfinite(k4)) {
            return y; // Return original value if integration fails
        }
        
        double result = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        
        // Final clamp and validity check
        result = std::max(0.0, std::min(result, 1.0));
        return std::isfinite(result) ? result : y;
    }
};

/**
 * @brief Fast sodium channel (Na_v)
 */
class SodiumChannel : public IonChannel {
private:
    double g_max_;    // Maximum conductance (mS/cm²)
    double E_Na_;     // Reversal potential (mV)
    double m_, h_;    // Gating variables
    
public:
    SodiumChannel(double g_max = 120.0, double E_Na = 50.0);
    double calculateCurrent(double V, double dt) override;
    double getConductance() const override;
    double getReversalPotential() const override;
    void reset() override;
    std::string getName() const override;
};

/**
 * @brief Delayed rectifier potassium channel (K_DR)
 */
class PotassiumChannel : public IonChannel {
private:
    double g_max_;    // Maximum conductance (mS/cm²)
    double E_K_;      // Reversal potential (mV)
    double n_;        // Gating variable
    
public:
    PotassiumChannel(double g_max = 36.0, double E_K = -77.0);
    double calculateCurrent(double V, double dt) override;
    double getConductance() const override;
    double getReversalPotential() const override;
    void reset() override;
    std::string getName() const override;
};

/**
 * @brief L-type calcium channel
 */
class CalciumChannel : public IonChannel {
private:
    double g_max_;    // Maximum conductance (mS/cm²)
    double E_Ca_;     // Reversal potential (mV)
    double m_, h_;    // Gating variables
    
public:
    CalciumChannel(double g_max = 0.5, double E_Ca = 120.0);
    double calculateCurrent(double V, double dt) override;
    double getConductance() const override;
    double getReversalPotential() const override;
    void reset() override;
    std::string getName() const override;
};

/**
 * @brief HCN hyperpolarization-activated channel
 */
class HCNChannel : public IonChannel {
private:
    double g_max_;    // Maximum conductance (mS/cm²)
    double E_h_;      // Reversal potential (mV)
    double m_;        // Gating variable
    
public:
    HCNChannel(double g_max = 0.1, double E_h = -30.0);
    double calculateCurrent(double V, double dt) override;
    double getConductance() const override;
    double getReversalPotential() const override;
    void reset() override;
    std::string getName() const override;
};

/**
 * @brief Abstract base class for synaptic receptors
 */
class SynapticReceptor {
public:
    virtual ~SynapticReceptor() = default;
    
    /**
     * @brief Process synaptic input and calculate current
     * @param V Membrane voltage (mV)
     * @param spike_times Vector of presynaptic spike times
     * @param current_time Current simulation time (ms)
     * @param dt Time step (ms)
     * @return Synaptic current (µA/cm²)
     */
    virtual double calculateCurrent(double V, const std::vector<double>& spike_times,
                                  double current_time, double dt) = 0;
    
    virtual std::string getName() const = 0;
    virtual void reset() = 0;
};

/**
 * @brief AMPA receptor implementation
 */
class AMPAReceptor : public SynapticReceptor {
private:
    double g_max_;      // Maximum conductance (mS/cm²)
    double E_syn_;      // Reversal potential (mV)
    double tau_rise_;   // Rise time constant (ms)
    double tau_decay_;  // Decay time constant (ms)
    double conductance_; // Current conductance state
    
public:
    AMPAReceptor(double g_max = 0.5, double E_syn = 0.0, 
                 double tau_rise = 0.2, double tau_decay = 2.0);
    double calculateCurrent(double V, const std::vector<double>& spike_times,
                          double current_time, double dt) override;
    std::string getName() const override;
    void reset() override;
};

/**
 * @brief NMDA receptor with voltage-dependent Mg²⁺ block
 */
class NMDAReceptor : public SynapticReceptor {
private:
    double g_max_;      // Maximum conductance (mS/cm²)
    double E_syn_;      // Reversal potential (mV)
    double tau_rise_;   // Rise time constant (ms)
    double tau_decay_;  // Decay time constant (ms)
    double conductance_; // Current conductance state
    
public:
    NMDAReceptor(double g_max = 0.1, double E_syn = 0.0,
                 double tau_rise = 2.0, double tau_decay = 100.0);
    double calculateCurrent(double V, const std::vector<double>& spike_times,
                          double current_time, double dt) override;
    std::string getName() const override;
    void reset() override;
};

/**
 * @brief GABA_A receptor implementation
 */
class GABAAReceptor : public SynapticReceptor {
private:
    double g_max_;      // Maximum conductance (mS/cm²)
    double E_syn_;      // Reversal potential (mV)
    double tau_rise_;   // Rise time constant (ms)
    double tau_decay_;  // Decay time constant (ms)
    double conductance_; // Current conductance state
    
public:
    GABAAReceptor(double g_max = 1.0, double E_syn = -70.0,
                  double tau_rise = 0.5, double tau_decay = 10.0);
    double calculateCurrent(double V, const std::vector<double>& spike_times,
                          double current_time, double dt) override;
    std::string getName() const override;
    void reset() override;
};

/**
 * @brief Individual compartment representing part of a neuron
 */
class Compartment {
public:
    enum Type {
        DENDRITE,
        SOMA,
        AIS,        // Axon Initial Segment
        AXON,
        TERMINAL
    };

private:
    Type type_;
    std::string name_;
    
    // Geometric parameters
    double length_;      // µm
    double diameter_;    // µm
    double area_;        // cm²
    
    // Electrical parameters
    double capacitance_; // µF/cm² (specific capacitance)
    double V_;           // Membrane voltage (mV)
    double axial_resistance_; // Ω·cm
    
    // Ion channels and synaptic receptors
    std::vector<std::unique_ptr<IonChannel>> ion_channels_;
    std::vector<std::unique_ptr<SynapticReceptor>> synaptic_receptors_;
    
    // Connectivity
    std::vector<std::shared_ptr<Compartment>> connected_compartments_;
    std::vector<double> coupling_conductances_; // mS
    
    // Synaptic input storage
    std::vector<std::vector<double>> synaptic_inputs_; // spike times for each receptor
    
    // Calcium dynamics
    double Ca_concentration_; // µM
    double Ca_target_;        // µM (resting concentration)
    double Ca_tau_;           // ms (decay time constant)
    double Ca_current_factor_; // Factor to convert Ca current to concentration change
    
public:
    Compartment(Type type, const std::string& name, double length, double diameter,
                double capacitance = 1.0, double axial_resistance = 150.0);
    
    void addIonChannel(std::unique_ptr<IonChannel> channel);
    void addSynapticReceptor(std::unique_ptr<SynapticReceptor> receptor);
    void connectTo(std::shared_ptr<Compartment> other, double conductance);
    void addSynapticInput(size_t receptor_idx, double spike_time);
    void updateVoltage(double I_external, double current_time, double dt);
    void cleanupSynapticInputs(double current_time, double cleanup_window = 500.0);
    void reset();
    
    // Getters
    double getVoltage() const { return V_; }
    void setVoltage(double V) { V_ = V; }
    Type getType() const { return type_; }
    const std::string& getName() const { return name_; }
    double getArea() const { return area_; }
    double getCapacitance() const { return capacitance_; }
    
private:
    double calculateTotalCurrent(double current_time, double dt) const;
    double calculateTotalCurrentAtVoltage(double V_test, double current_time, double dt);
};

/**
 * @brief Main neuron class containing multiple compartments
 */
class Neuron {
private:
    std::string id_;
    std::vector<std::shared_ptr<Compartment>> compartments_;
    std::unordered_map<std::string, std::shared_ptr<Compartment>> compartment_map_;
    NeuronConfig config_;
    
    // Simulation state
    double current_time_;
    std::vector<double> spike_times_;
    double last_spike_time_;
    double spike_threshold_;
    
public:
    Neuron(const std::string& id, const NeuronConfig& config = NeuronConfig());
    
    void addCompartment(std::shared_ptr<Compartment> compartment);
    std::shared_ptr<Compartment> getCompartment(const std::string& name);
    void createStandardMorphology();
    void step(double I_external = 0.0, double dt_param = -1.0);
    void addSynapticInput(const std::string& compartment_name, 
                         size_t receptor_idx, double spike_time);
    void simulate(double duration, double I_external = 0.0);
    void reset();
    bool loadConfig(const std::string& filename);
    
    // Getters
    const std::vector<double>& getSpikeTimes() const { return spike_times_; }
    double getCurrentTime() const { return current_time_; }
    const std::string& getId() const { return id_; }
    double getCompartmentVoltage(const std::string& name) const;
    void printState() const;
    double getFiringRate(double time_window) const;
    double getLastSpikeTime() const { return last_spike_time_; }
    
    // Methods needed by Network class
    void update(double dt);
    bool hasFired() const { 
        return !spike_times_.empty() && 
               (current_time_ - spike_times_.back()) < config_.dt * 2.0; 
    }
    void receiveSynapticInput(double weight) { 
        addSynapticInput("soma", 0, current_time_); 
        (void)weight; // Suppress unused parameter warning
    }
    void injectCurrent(double current) { 
        step(current); 
    }
    void receiveNeuromodulator(const std::string& type, double concentration) {
        // Placeholder for neuromodulator effects
        (void)type; (void)concentration; // Suppress unused parameter warnings
    }
};

/**
 * @brief Network class for managing multiple neurons with Hebbian learning
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::unordered_map<std::string, size_t> neuron_indices_;
    
    // Hebbian learning parameters
    struct HebbianConnection {
        size_t pre_neuron, post_neuron;
        std::string pre_compartment, post_compartment;
        size_t receptor_idx;
        double weight;
        double learning_rate;
        double decay_rate;
    };
    
    std::vector<HebbianConnection> connections_;
    
public:
    void addNeuron(std::unique_ptr<Neuron> neuron);
    void createHebbianConnection(const std::string& pre_id, const std::string& post_id,
                               const std::string& post_compartment, size_t receptor_idx,
                               double initial_weight = 0.1, double learning_rate = 0.01);
    void updateHebbianWeights(double time_window = 20.0);
    void propagateSpikes();
    void step(const std::vector<double>& external_currents = {});
    Neuron* getNeuron(const std::string& id);
    void printNetworkStats() const;
};

#endif // NEURON_H