#include "Neuron.h"
#include "Network.h"
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

// ============================================================================
// GATING HELPER STRUCTURES AND FUNCTIONS
// ============================================================================

struct GatingRates {
    double alpha, beta;
};

// ============================================================================
// ION CHANNEL BASE CLASS IMPLEMENTATION
// ============================================================================


// ============================================================================
// ION CHANNEL METHOD IMPLEMENTATIONS
// ============================================================================

// SodiumChannel implementations
SodiumChannel::SodiumChannel(double g_max, double E_Na)
    : g_max_(g_max), E_Na_(E_Na), m_(0.0529), h_(0.5961) {}

double SodiumChannel::calculateCurrent(double V, double dt) {
    if (!std::isfinite(V)) {
        std::cerr << "Warning: Invalid voltage in SodiumChannel: " << V << std::endl;
        return 0.0;
    }
    
    m_ = integrateGating(m_, V, dt, [](double V) -> GatingRates {
        double denom = (V + 40.0);
        if (std::abs(denom) < 1e-6) denom = 1e-6;
        
        double alpha_m = 0.1 * denom / (1.0 - std::exp(-denom / 10.0));
        double beta_m = 4.0 * std::exp(-(V + 65.0) / 18.0);
        
        alpha_m = std::max(0.0, std::min(alpha_m, 1000.0));
        beta_m = std::max(0.0, std::min(beta_m, 1000.0));
        
        return {alpha_m, beta_m};
    });
    
    h_ = integrateGating(h_, V, dt, [](double V) -> GatingRates {
        double alpha_h = 0.07 * std::exp(-(V + 65.0) / 20.0);
        double beta_h = 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
        
        alpha_h = std::max(0.0, std::min(alpha_h, 1000.0));
        beta_h = std::max(0.0, std::min(beta_h, 1000.0));
        
        return {alpha_h, beta_h};
    });
    
    m_ = std::max(0.0, std::min(m_, 1.0));
    h_ = std::max(0.0, std::min(h_, 1.0));
    
    double g = g_max_ * std::pow(m_, 3) * h_;
    double current = g * (V - E_Na_);
    
    if (!std::isfinite(current)) {
        std::cerr << "Warning: Invalid current in SodiumChannel: " << current << std::endl;
        return 0.0;
    }
    
    return current;
}

double SodiumChannel::getConductance() const { return g_max_; }
double SodiumChannel::getReversalPotential() const { return E_Na_; }
void SodiumChannel::reset() { m_ = 0.0529; h_ = 0.5961; }
std::string SodiumChannel::getName() const { return "Na_v"; }

// PotassiumChannel implementations
PotassiumChannel::PotassiumChannel(double g_max, double E_K)
    : g_max_(g_max), E_K_(E_K), n_(0.3177) {}

double PotassiumChannel::calculateCurrent(double V, double dt) {
    if (!std::isfinite(V)) {
        return 0.0;
    }
    
    n_ = integrateGating(n_, V, dt, [](double V) -> GatingRates {
        double denom = (V + 55.0);
        if (std::abs(denom) < 1e-6) denom = 1e-6;
        
        double alpha_n = 0.01 * denom / (1.0 - std::exp(-denom / 10.0));
        double beta_n = 0.125 * std::exp(-(V + 65.0) / 80.0);
        
        alpha_n = std::max(0.0, std::min(alpha_n, 1000.0));
        beta_n = std::max(0.0, std::min(beta_n, 1000.0));
        
        return {alpha_n, beta_n};
    });
    
    n_ = std::max(0.0, std::min(n_, 1.0));
    
    double g = g_max_ * std::pow(n_, 4);
    double current = g * (V - E_K_);
    
    return std::isfinite(current) ? current : 0.0;
}

double PotassiumChannel::getConductance() const { return g_max_; }
double PotassiumChannel::getReversalPotential() const { return E_K_; }
void PotassiumChannel::reset() { n_ = 0.3177; }
std::string PotassiumChannel::getName() const { return "K_DR"; }

// CalciumChannel implementations
CalciumChannel::CalciumChannel(double g_max, double E_Ca)
    : g_max_(g_max), E_Ca_(E_Ca), m_(0.0), h_(1.0) {}

double CalciumChannel::calculateCurrent(double V, double dt) {
    m_ = integrateGating(m_, V, dt, [](double V) -> GatingRates {
        double alpha_m = 0.055 * (V + 27.0) / (1.0 - std::exp(-(V + 27.0) / 3.8));
        double beta_m = 0.94 * std::exp(-(V + 75.0) / 17.0);
        return {alpha_m, beta_m};
    });
    
    h_ = integrateGating(h_, V, dt, [](double V) -> GatingRates {
        double alpha_h = 0.000457 * std::exp(-(V + 13.0) / 50.0);
        double beta_h = 0.0065 / (1.0 + std::exp(-(V + 15.0) / 28.0));
        return {alpha_h, beta_h};
    });
    
    double g = g_max_ * m_ * h_;
    return g * (V - E_Ca_);
}

double CalciumChannel::getConductance() const { return g_max_; }
double CalciumChannel::getReversalPotential() const { return E_Ca_; }
void CalciumChannel::reset() { m_ = 0.0; h_ = 1.0; }
std::string CalciumChannel::getName() const { return "Ca_L"; }

// HCNChannel implementations  
HCNChannel::HCNChannel(double g_max, double E_h)
    : g_max_(g_max), E_h_(E_h), m_(0.0) {}

double HCNChannel::calculateCurrent(double V, double dt) {
    m_ = integrateGating(m_, V, dt, [](double V) -> GatingRates {
        double alpha_m = 0.001 / (1.0 + std::exp((V + 90.0) / 10.0));
        double beta_m = 0.001 / (1.0 + std::exp(-(V + 90.0) / 10.0));
        return {alpha_m, beta_m};
    });
    
    double g = g_max_ * m_;
    return g * (V - E_h_);
}

double HCNChannel::getConductance() const { return g_max_; }
double HCNChannel::getReversalPotential() const { return E_h_; }
void HCNChannel::reset() { m_ = 0.0; }
std::string HCNChannel::getName() const { return "HCN"; }

// ============================================================================
// SYNAPTIC RECEPTOR METHOD IMPLEMENTATIONS
// ============================================================================

// AMPAReceptor implementations
AMPAReceptor::AMPAReceptor(double g_max, double E_syn, double tau_rise, double tau_decay)
    : g_max_(g_max), E_syn_(E_syn), tau_rise_(tau_rise), 
      tau_decay_(tau_decay), conductance_(0.0) {}

double AMPAReceptor::calculateCurrent(double V, const std::vector<double>& spike_times,
                      double current_time, double dt) {
    conductance_ *= std::exp(-dt / tau_decay_);
    
    for (double spike_time : spike_times) {
        double delta_t = current_time - spike_time;
        if (delta_t >= 0.0 && delta_t <= 5.0 * tau_decay_) {
            double weight = std::exp(-delta_t / tau_decay_) - std::exp(-delta_t / tau_rise_);
            conductance_ += g_max_ * weight;
        }
    }
    
    return conductance_ * (V - E_syn_);
}

std::string AMPAReceptor::getName() const { return "AMPA"; }
void AMPAReceptor::reset() { conductance_ = 0.0; }

// NMDAReceptor implementations
NMDAReceptor::NMDAReceptor(double g_max, double E_syn, double tau_rise, double tau_decay)
    : g_max_(g_max), E_syn_(E_syn), tau_rise_(tau_rise),
      tau_decay_(tau_decay), conductance_(0.0) {}

double NMDAReceptor::calculateCurrent(double V, const std::vector<double>& spike_times,
                      double current_time, double dt) {
    conductance_ *= std::exp(-dt / tau_decay_);
    
    for (double spike_time : spike_times) {
        double delta_t = current_time - spike_time;
        if (delta_t >= 0.0 && delta_t <= 5.0 * tau_decay_) {
            double weight = std::exp(-delta_t / tau_decay_) - std::exp(-delta_t / tau_rise_);
            conductance_ += g_max_ * weight;
        }
    }
    
    double mg_block = 1.0 / (1.0 + 0.28 * std::exp(-0.062 * V));
    
    return conductance_ * mg_block * (V - E_syn_);
}

std::string NMDAReceptor::getName() const { return "NMDA"; }
void NMDAReceptor::reset() { conductance_ = 0.0; }

// GABAAReceptor implementations
GABAAReceptor::GABAAReceptor(double g_max, double E_syn, double tau_rise, double tau_decay)
    : g_max_(g_max), E_syn_(E_syn), tau_rise_(tau_rise),
      tau_decay_(tau_decay), conductance_(0.0) {}

double GABAAReceptor::calculateCurrent(double V, const std::vector<double>& spike_times,
                      double current_time, double dt) {
    conductance_ *= std::exp(-dt / tau_decay_);
    
    for (double spike_time : spike_times) {
        double delta_t = current_time - spike_time;
        if (delta_t >= 0.0 && delta_t <= 5.0 * tau_decay_) {
            double weight = std::exp(-delta_t / tau_decay_) - std::exp(-delta_t / tau_rise_);
            conductance_ += g_max_ * weight;
        }
    }
    
    return conductance_ * (V - E_syn_);
}

std::string GABAAReceptor::getName() const { return "GABA_A"; }
void GABAAReceptor::reset() { conductance_ = 0.0; }

// ============================================================================
// NEURON CLASS METHOD IMPLEMENTATIONS
// ============================================================================

// Helper function to create compartment
std::shared_ptr<Compartment> createCompartment(Compartment::Type type, const std::string& name, 
                                              double length, double diameter, double capacitance, double axial_resistance) {
    return std::make_shared<Compartment>(type, name, length, diameter, capacitance, axial_resistance);
}

Neuron::Neuron(const std::string& id, const NeuronConfig& config)
    : id_(id), config_(config), current_time_(0.0), last_spike_time_(-1000.0),
      spike_threshold_(-20.0) {}

void Neuron::addCompartment(std::shared_ptr<Compartment> compartment) {
    compartments_.push_back(compartment);
    compartment_map_[compartment->getName()] = compartment;
}

std::shared_ptr<Compartment> Neuron::getCompartment(const std::string& name) {
    auto it = compartment_map_.find(name);
    return (it != compartment_map_.end()) ? it->second : nullptr;
}

void Neuron::createStandardMorphology() {
    // Create compartments using helper
    auto dendrite = createCompartment(
        Compartment::DENDRITE, "dendrite", 100.0, 2.0, 
        config_.compartment_defaults.capacitance, 150.0);
    auto soma = createCompartment(
        Compartment::SOMA, "soma", 20.0, 20.0,
        config_.compartment_defaults.capacitance, 150.0);
    auto ais = createCompartment(
        Compartment::AIS, "ais", 25.0, 1.0,
        config_.compartment_defaults.capacitance, 150.0);
    auto axon = createCompartment(
        Compartment::AXON, "axon", 1000.0, 1.0,
        config_.compartment_defaults.capacitance, 150.0);
    
    // Add ion channels to each compartment
    // Dendrite: NMDA, AMPA, GABA, some Ca channels
    dendrite->addIonChannel(std::make_unique<CalciumChannel>(0.2));
    dendrite->addSynapticReceptor(std::make_unique<AMPAReceptor>());
    dendrite->addSynapticReceptor(std::make_unique<NMDAReceptor>());
    dendrite->addSynapticReceptor(std::make_unique<GABAAReceptor>());
    
    // Soma: Standard HH channels + synaptic receptors
    soma->addIonChannel(std::make_unique<SodiumChannel>(50.0));
    soma->addIonChannel(std::make_unique<PotassiumChannel>(20.0));
    soma->addIonChannel(std::make_unique<CalciumChannel>(0.5));
    soma->addSynapticReceptor(std::make_unique<AMPAReceptor>());
    soma->addSynapticReceptor(std::make_unique<GABAAReceptor>());
    
    // AIS: High density Na and K channels for action potential initiation
    ais->addIonChannel(std::make_unique<SodiumChannel>(200.0));
    ais->addIonChannel(std::make_unique<PotassiumChannel>(80.0));
    
    // Axon: Standard propagation channels
    axon->addIonChannel(std::make_unique<SodiumChannel>(120.0));
    axon->addIonChannel(std::make_unique<PotassiumChannel>(36.0));
    
    // Connect compartments
    dendrite->connectTo(soma, 1.0);   // mS coupling
    soma->connectTo(dendrite, 1.0);
    soma->connectTo(ais, 2.0);
    ais->connectTo(soma, 2.0);
    ais->connectTo(axon, 1.5);
    axon->connectTo(ais, 1.5);
    
    // Add to neuron
    addCompartment(dendrite);
    addCompartment(soma);
    addCompartment(ais);
    addCompartment(axon);
}

void Neuron::step(double I_external, double dt_param) {
    double dt_to_use = (dt_param > 0 && std::isfinite(dt_param)) ? dt_param : config_.dt;

    // Update all compartments
    for (auto& compartment_pair : compartment_map_) {
        auto& compartment = compartment_pair.second;
        double I_ext_comp = (compartment->getName() == "soma" || compartment->getType() == Compartment::SOMA) ? I_external : 0.0;
        compartment->updateVoltage(I_ext_comp, current_time_, dt_to_use);
    }
    
    // Check for spike in soma (or AIS if defined as primary spike zone)
    auto spike_source_compartment = getCompartment("soma"); // Default to soma
    if (!spike_source_compartment) {
         auto ais_it = compartment_map_.find("ais");
         if (ais_it != compartment_map_.end()) {
            spike_source_compartment = ais_it->second;
         }
    }

    if (spike_source_compartment && spike_source_compartment->getVoltage() > spike_threshold_ && 
        (current_time_ - last_spike_time_) > 2.0) { // 2ms refractory period
        spike_times_.push_back(current_time_);
        last_spike_time_ = current_time_;
    }
    
    // Clean up old synaptic inputs periodically (every 10ms)
    if (fmod(current_time_, 10.0) < dt_to_use && fmod(current_time_, 10.0) >= 0.0) { 
        for (auto& compartment_pair : compartment_map_) {
            compartment_pair.second->cleanupSynapticInputs(current_time_);
        }
    }
    
    current_time_ += dt_to_use;
}

void Neuron::addSynapticInput(const std::string& compartment_name, 
                             size_t receptor_idx, double spike_time) {
    auto comp = getCompartment(compartment_name);
    if (comp) {
        comp->addSynapticInput(receptor_idx, spike_time);
    }
}

void Neuron::simulate(double duration, double I_external) {
    double end_time = current_time_ + duration;
    while (current_time_ < end_time) {
        step(I_external);
    }
}

void Neuron::reset() {
    current_time_ = 0.0;
    last_spike_time_ = -1000.0;
    spike_times_.clear();
    
    for (auto& compartment : compartments_) {
        compartment->reset();
    }
}

bool Neuron::loadConfig(const std::string& /*filename*/) {
    // JSON support disabled for now
    return true;
}

double Neuron::getCompartmentVoltage(const std::string& name) const {
    auto it = compartment_map_.find(name);
    return (it != compartment_map_.end()) ? it->second->getVoltage() : -1000.0;
}

void Neuron::printState() const {
    std::cout << "Neuron " << id_ << " at t=" << current_time_ << "ms:\n";
    for (const auto& comp_pair : compartment_map_) {
        std::cout << "  " << comp_pair.first << ": " 
                    << comp_pair.second->getVoltage() << " mV\n";
    }
    std::cout << "  Spikes: " << spike_times_.size() << std::endl;
}

double Neuron::getFiringRate(double time_window) const {
    if (time_window <= 0) return 0.0;
    double window_start_time = current_time_ - time_window;
    int spike_count = 0;
    for (double spike_time : spike_times_) {
        if (spike_time >= window_start_time) {
            spike_count++;
        }
    }
    return static_cast<double>(spike_count) / time_window * 1000.0; // Spikes per second
}

void Neuron::update(double dt) {
    step(0.0, dt);
}

// ============================================================================
// NEURONFACTORY CLASS METHOD IMPLEMENTATIONS
// ============================================================================

std::unordered_map<std::string, std::function<std::shared_ptr<Neuron>(const std::string&, const Position3D&)>> 
    NeuronFactory::custom_creators_;

std::shared_ptr<Neuron> NeuronFactory::createNeuron(NeuronType type, 
                                                   const std::string& id, 
                                                   const Position3D& position) {
    NeuronConfig config;
    
    // Modify config based on neuron type
    switch (type) {
        case PYRAMIDAL_CORTICAL:
            config.compartment_defaults.capacitance = 1.0;
            config.dt = 0.01;
            break;
        case INTERNEURON_BASKET:
            config.compartment_defaults.capacitance = 0.8;
            config.dt = 0.01;
            break;
        case INTERNEURON_CHANDELIER:
            config.compartment_defaults.capacitance = 0.6;
            config.dt = 0.01;
            break;
        case PURKINJE_CEREBELLAR:
            config.compartment_defaults.capacitance = 2.0;
            config.dt = 0.005; // Faster time step for complex dynamics
            break;
        case GRANULE_CEREBELLAR:
            config.compartment_defaults.capacitance = 0.3;
            config.dt = 0.01;
            break;
        case SENSORY:
            config.compartment_defaults.capacitance = 0.9;
            config.dt = 0.01;
            break;
        case MOTOR:
            config.compartment_defaults.capacitance = 1.2;
            config.dt = 0.01;
            break;
        default:
            break;
    }
    
    auto neuron = std::make_shared<Neuron>(id, config);
    neuron->createStandardMorphology();
    
    // Position is stored in Network class, not in Neuron
    // The position parameter is used by the Network for spatial organization
    (void)position; // Silence unused parameter warning
    
    return neuron;
}

void NeuronFactory::registerCustomType(const std::string& type_name,
                                      std::function<std::shared_ptr<Neuron>(const std::string&, const Position3D&)> creator) {
    custom_creators_[type_name] = creator;
}

// ===================== Compartment Implementation =====================
Compartment::Compartment(Type type, const std::string& name, double length, double diameter, double capacitance, double axial_resistance)
    : type_(type), name_(name), length_(length), diameter_(diameter), capacitance_(capacitance), V_(-65.0), axial_resistance_(axial_resistance) {
    area_ = M_PI * diameter_ * length_ * 1e-8;
    if (area_ <= 0) area_ = 1e-8;
}

void Compartment::addIonChannel(std::unique_ptr<IonChannel> channel) {
    ion_channels_.push_back(std::move(channel));
}

void Compartment::addSynapticReceptor(std::unique_ptr<SynapticReceptor> receptor) {
    synaptic_receptors_.push_back(std::move(receptor));
    synaptic_inputs_.emplace_back(); // Initialize empty spike time vector
}

void Compartment::connectTo(std::shared_ptr<Compartment> other, double conductance) {
    connected_compartments_.push_back(other);
    coupling_conductances_.push_back(conductance);
}

void Compartment::addSynapticInput(size_t receptor_idx, double spike_time) {
    if (receptor_idx < synaptic_inputs_.size()) {
        synaptic_inputs_[receptor_idx].push_back(spike_time);
    }
}

void Compartment::updateVoltage(double I_external, double current_time, double dt) {
    // Safety check for input parameters
    if (!std::isfinite(I_external) || !std::isfinite(current_time) || !std::isfinite(dt) || dt <= 0) {
        return;
    }
    
    // RK4 integration for membrane voltage
    auto dVdt_lambda = [this, I_external, current_time, dt](double V_temp) -> double {
        V_temp = std::max(-150.0, std::min(V_temp, 100.0));
        double I_calculated_ionic_and_synaptic = this->calculateTotalCurrentAtVoltage(V_temp, current_time, dt); 
        if (!std::isfinite(I_calculated_ionic_and_synaptic)) {
            return 0.0;
        }
        double C_m_total = this->capacitance_ * this->area_;
        if (C_m_total <= 1e-12) {
             return 0.0; 
        }
        double dv = (I_external + I_calculated_ionic_and_synaptic) / C_m_total;
        return std::max(-1000.0, std::min(dv, 1000.0)); 
    };
    
    double k1 = dt * dVdt_lambda(V_);
    double k2 = dt * dVdt_lambda(V_ + k1/2.0);
    double k3 = dt * dVdt_lambda(V_ + k2/2.0);
    double k4 = dt * dVdt_lambda(V_ + k3);
    
    // Check all k values are finite
    if (std::isfinite(k1) && std::isfinite(k2) && std::isfinite(k3) && std::isfinite(k4)) {
        double dV = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
        
        // Limit voltage change per step
        dV = std::max(-10.0, std::min(dV, 10.0)); // Max 10mV change per step
        
        V_ += dV;
        
        // Clamp voltage to physiological range
        V_ = std::max(-150.0, std::min(V_, 100.0));
    }
    
    // Now update channel and receptor states with the actual dt
    calculateTotalCurrent(current_time, dt);
}

void Compartment::cleanupSynapticInputs(double current_time, double cleanup_window) {
    for (auto& input_list : synaptic_inputs_) {
        auto it = std::remove_if(input_list.begin(), input_list.end(),
            [current_time, cleanup_window](double spike_time) -> bool {
                return (current_time - spike_time) > cleanup_window;
            });
        input_list.erase(it, input_list.end());
    }
}

void Compartment::reset() {
    V_ = -65.0;
    for (auto& channel : ion_channels_) {
        channel->reset();
    }
    for (auto& receptor : synaptic_receptors_) {
        receptor->reset();
    }
    for (auto& input_list : synaptic_inputs_) {
        input_list.clear();
    }
}

double Compartment::calculateTotalCurrent(double current_time, double dt) const {
    double I_total = 0.0;
    
    // Ion channel currents
    for (const auto& channel : ion_channels_) {
        I_total += const_cast<IonChannel*>(channel.get())->calculateCurrent(V_, dt);
    }
    
    // Synaptic currents
    for (size_t i = 0; i < synaptic_receptors_.size(); ++i) {
        if (i < synaptic_inputs_.size()) { 
            I_total += const_cast<SynapticReceptor*>(synaptic_receptors_[i].get())->calculateCurrent(
                V_, synaptic_inputs_[i], current_time, dt);
        }
    }
    
    // Axial currents from connected compartments
    for (size_t i = 0; i < connected_compartments_.size(); ++i) {
        double V_other = connected_compartments_[i]->getVoltage();
        I_total += coupling_conductances_[i] * (V_other - V_);
    }
    
    return I_total;
}

double Compartment::calculateTotalCurrentAtVoltage(double voltage, double current_time, double dt) {
    double total_current = 0.0;
    
    // Ion channel currents
    for (auto& channel : ion_channels_) {
        total_current -= channel->calculateCurrent(voltage, dt);
    }
    
    // Synaptic currents
    for (size_t i = 0; i < synaptic_receptors_.size(); ++i) {
        if (i < synaptic_inputs_.size()) { 
            total_current -= synaptic_receptors_[i]->calculateCurrent(voltage, synaptic_inputs_[i], current_time, dt);
        }
    }
    
    return total_current;
}
