# Phase 4: Neuromodulation System

## Overview

The neuromodulatory system is almost entirely missing from the current implementation. This phase will implement a comprehensive neuromodulatory system that regulates network states, influences learning, and enables adaptive responses to changing market conditions.

## Current Implementation Analysis

From the code analysis, we found:

- No dedicated neuromodulatory system exists
- Basic reward signals are used but without proper neuromodulatory effects
- Network states are not regulated by neuromodulators
- Learning is not modulated by different neuromodulatory signals

## Implementation Tasks

### 1. Create Neuromodulator Manager

#### 1.1 Define Neuromodulator Structure

```cpp
// NeuromodulatorSystem.h
#ifndef NEUROMODULATOR_SYSTEM_H
#define NEUROMODULATOR_SYSTEM_H

#include <cuda_runtime.h>
#include "NeuronModelConstants.h"
#include "LearningRuleConstants.h"

// Neuromodulator types
enum NeuromodulatorType {
    DOPAMINE = 0,
    SEROTONIN = 1,
    ACETYLCHOLINE = 2,
    NORADRENALINE = 3,
    NUM_NEUROMODULATORS
};

// Structure to hold global neuromodulator levels
struct GlobalNeuromodulators {
    float levels[NUM_NEUROMODULATORS];
    float baseline[NUM_NEUROMODULATORS];
    float decay_rates[NUM_NEUROMODULATORS];
    float production_rates[NUM_NEUROMODULATORS];
    float min_levels[NUM_NEUROMODULATORS];
    float max_levels[NUM_NEUROMODULATORS];
    float current_production[NUM_NEUROMODULATORS];
};

// Structure to hold local neuromodulator levels for each neuron
struct LocalNeuromodulators {
    float dopamine;
    float serotonin;
    float acetylcholine;
    float noradrenaline;
    
    // Receptors for each neuromodulator (sensitivity)
    float dopamine_receptors;
    float serotonin_receptors;
    float acetylcholine_receptors;
    float noradrenaline_receptors;
    
    // Receptor desensitization state
    float dopamine_desensitization;
    float serotonin_desensitization;
    float acetylcholine_desensitization;
    float noradrenaline_desensitization;
};

// Neuromodulator source neuron
struct NeuromodulatorSource {
    int neuron_idx;
    NeuromodulatorType type;
    float release_rate;
    float release_threshold;
    float current_release;
    float recovery_rate;
    float max_release;
};

#endif // NEUROMODULATOR_SYSTEM_H
```

#### 1.2 Update GPUNeuronState Structure

```cpp
struct GPUNeuronState {
    // Existing fields from previous phases
    // ...
    
    // Neuromodulator-related fields
    LocalNeuromodulators neuromodulators;
    int neuron_type;                  // Regular, dopaminergic, serotonergic, etc.
    float neuromodulator_release;     // Amount of neuromodulator released when spiking
    float neuromodulator_threshold;   // Threshold for neuromodulator release
    
    // Neuromodulator sensitivity
    float dopamine_sensitivity;
    float serotonin_sensitivity;
    float acetylcholine_sensitivity;
    float noradrenaline_sensitivity;
    
    // State regulation
    float excitability_modulation;    // Current modulation of excitability
    float plasticity_modulation;      // Current modulation of plasticity
    float adaptation_rate;            // Rate of adaptation to neuromodulators
};
```

#### 1.3 Create Global Neuromodulator Manager

```cpp
// NeuromodulatorManager.h
#ifndef NEUROMODULATOR_MANAGER_H
#define NEUROMODULATOR_MANAGER_H

#include "NeuromodulatorSystem.h"
#include <vector>

class NeuromodulatorManager {
private:
    // Host-side data
    GlobalNeuromodulators h_global_neuromodulators;
    std::vector<NeuromodulatorSource> h_sources;
    
    // Device-side data
    GlobalNeuromodulators* d_global_neuromodulators;
    NeuromodulatorSource* d_sources;
    int num_sources;
    
    // Market state tracking for neuromodulator production
    float market_volatility;
    float market_trend;
    float profit_loss;
    float prediction_error;
    
    // Internal methods
    void updateMarketState(float volatility, float trend, float pnl);
    void computeNeuromodulatorProduction();
    
public:
    NeuromodulatorManager();
    ~NeuromodulatorManager();
    
    // Initialization
    void initialize(int num_neurons);
    void registerSource(int neuron_idx, NeuromodulatorType type, float release_rate);
    
    // Runtime operations
    void update(float dt, float reward_signal);
    void processMarketEvent(float volatility, float trend, float pnl);
    void injectNeuromodulator(NeuromodulatorType type, float amount);
    
    // Getters
    float getNeuromodulatorLevel(NeuromodulatorType type) const;
    const GlobalNeuromodulators& getGlobalLevels() const;
    
    // CUDA kernel launchers
    void updateLocalNeuromodulators(GPUNeuronState* neurons, int N, float dt);
    void processNeuromodulatorRelease(GPUNeuronState* neurons, int N, float dt);
    void applyNeuromodulatorEffects(GPUNeuronState* neurons, GPUSynapse* synapses, 
                                   int N, int num_synapses, float dt);
};

#endif // NEUROMODULATOR_MANAGER_H
```

### 2. Implement Neuromodulator Dynamics

#### 2.1 Create Global Neuromodulator Update Kernel

```cpp
__global__ void updateGlobalNeuromodulatorsKernel(GlobalNeuromodulators* global_nm, 
                                                 float dt, float reward_signal) {
    // This kernel updates the global neuromodulator levels
    
    // Only one thread should execute this
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Update each neuromodulator
    for (int i = 0; i < NUM_NEUROMODULATORS; i++) {
        // Current level and parameters
        float level = global_nm->levels[i];
        float baseline = global_nm->baseline[i];
        float decay_rate = global_nm->decay_rates[i];
        float production = global_nm->current_production[i];
        
        // Special case for dopamine: directly affected by reward signal
        if (i == DOPAMINE) {
            production += reward_signal * REWARD_TO_DOPAMINE_FACTOR;
        }
        
        // Update level with production and decay
        float dlevel = production - decay_rate * (level - baseline);
        level += dlevel * dt;
        
        // Ensure level stays within bounds
        if (level < global_nm->min_levels[i]) {
            level = global_nm->min_levels[i];
        } else if (level > global_nm->max_levels[i]) {
            level = global_nm->max_levels[i];
        }
        
        // Store updated level
        global_nm->levels[i] = level;
    }
}
```

#### 2.2 Create Local Neuromodulator Update Kernel

```cpp
__global__ void updateLocalNeuromodulatorsKernel(GPUNeuronState* neurons, 
                                               GlobalNeuromodulators* global_nm,
                                               float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Get global levels
    float global_dopamine = global_nm->levels[DOPAMINE];
    float global_serotonin = global_nm->levels[SEROTONIN];
    float global_acetylcholine = global_nm->levels[ACETYLCHOLINE];
    float global_noradrenaline = global_nm->levels[NORADRENALINE];
    
    // Get local levels
    float local_dopamine = neuron.neuromodulators.dopamine;
    float local_serotonin = neuron.neuromodulators.serotonin;
    float local_acetylcholine = neuron.neuromodulators.acetylcholine;
    float local_noradrenaline = neuron.neuromodulators.noradrenaline;
    
    // Get receptor sensitivities (accounting for desensitization)
    float dopamine_sensitivity = neuron.dopamine_sensitivity * 
                               (1.0f - neuron.neuromodulators.dopamine_desensitization);
    float serotonin_sensitivity = neuron.serotonin_sensitivity * 
                                (1.0f - neuron.neuromodulators.serotonin_desensitization);
    float acetylcholine_sensitivity = neuron.acetylcholine_sensitivity * 
                                    (1.0f - neuron.neuromodulators.acetylcholine_desensitization);
    float noradrenaline_sensitivity = neuron.noradrenaline_sensitivity * 
                                    (1.0f - neuron.neuromodulators.noradrenaline_desensitization);
    
    // Diffusion from global to local (with sensitivity modulation)
    float dopamine_diffusion = (global_dopamine - local_dopamine) * dopamine_sensitivity;
    float serotonin_diffusion = (global_serotonin - local_serotonin) * serotonin_sensitivity;
    float acetylcholine_diffusion = (global_acetylcholine - local_acetylcholine) * acetylcholine_sensitivity;
    float noradrenaline_diffusion = (global_noradrenaline - local_noradrenaline) * noradrenaline_sensitivity;
    
    // Update local levels
    neuron.neuromodulators.dopamine += dopamine_diffusion * dt;
    neuron.neuromodulators.serotonin += serotonin_diffusion * dt;
    neuron.neuromodulators.acetylcholine += acetylcholine_diffusion * dt;
    neuron.neuromodulators.noradrenaline += noradrenaline_diffusion * dt;
    
    // Update receptor desensitization
    // Higher local levels lead to more desensitization
    neuron.neuromodulators.dopamine_desensitization += 
        (local_dopamine * DESENSITIZATION_RATE - neuron.neuromodulators.dopamine_desensitization * RESENSITIZATION_RATE) * dt;
    neuron.neuromodulators.serotonin_desensitization += 
        (local_serotonin * DESENSITIZATION_RATE - neuron.neuromodulators.serotonin_desensitization * RESENSITIZATION_RATE) * dt;
    neuron.neuromodulators.acetylcholine_desensitization += 
        (local_acetylcholine * DESENSITIZATION_RATE - neuron.neuromodulators.acetylcholine_desensitization * RESENSITIZATION_RATE) * dt;
    neuron.neuromodulators.noradrenaline_desensitization += 
        (local_noradrenaline * DESENSITIZATION_RATE - neuron.neuromodulators.noradrenaline_desensitization * RESENSITIZATION_RATE) * dt;
    
    // Clamp desensitization values
    neuron.neuromodulators.dopamine_desensitization = fminf(fmaxf(neuron.neuromodulators.dopamine_desensitization, 0.0f), 1.0f);
    neuron.neuromodulators.serotonin_desensitization = fminf(fmaxf(neuron.neuromodulators.serotonin_desensitization, 0.0f), 1.0f);
    neuron.neuromodulators.acetylcholine_desensitization = fminf(fmaxf(neuron.neuromodulators.acetylcholine_desensitization, 0.0f), 1.0f);
    neuron.neuromodulators.noradrenaline_desensitization = fminf(fmaxf(neuron.neuromodulators.noradrenaline_desensitization, 0.0f), 1.0f);
}
```

#### 2.3 Implement Neuromodulator Release Kernel

```cpp
__global__ void neuromodulatorReleaseKernel(GPUNeuronState* neurons, 
                                          NeuromodulatorSource* sources,
                                          GlobalNeuromodulators* global_nm,
                                          int num_sources, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sources) return;
    
    NeuromodulatorSource& source = sources[idx];
    int neuron_idx = source.neuron_idx;
    
    // Check if source neuron is active
    if (neurons[neuron_idx].active == 0) return;
    
    // Check if neuron has spiked
    bool has_spiked = neurons[neuron_idx].spiked;
    float voltage = neurons[neuron_idx].voltage;
    
    // Determine release amount
    float release = 0.0f;
    
    if (has_spiked) {
        // Spike-triggered release
        release = source.release_rate * source.current_release;
        
        // Deplete available neuromodulator
        source.current_release *= (1.0f - RELEASE_DEPLETION_FACTOR);
    } else if (voltage > source.release_threshold) {
        // Voltage-dependent release (subthreshold)
        float voltage_factor = (voltage - source.release_threshold) / 
                              (SPIKE_THRESHOLD - source.release_threshold);
        release = source.release_rate * source.current_release * voltage_factor * SUBTHRESHOLD_RELEASE_FACTOR;
        
        // Deplete available neuromodulator (less than spike-triggered)
        source.current_release *= (1.0f - RELEASE_DEPLETION_FACTOR * voltage_factor);
    }
    
    // Recovery of releasable neuromodulator
    source.current_release += (source.max_release - source.current_release) * source.recovery_rate * dt;
    
    // Add release to global level
    if (release > 0.0f) {
        atomicAdd(&global_nm->levels[source.type], release);
    }
}
```

### 3. Implement Neuromodulatory Effects

#### 3.1 Create Neuron Modulation Kernel

```cpp
__global__ void applyNeuromodulatorEffectsKernel(GPUNeuronState* neurons, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Get local neuromodulator levels
    float dopamine = neuron.neuromodulators.dopamine;
    float serotonin = neuron.neuromodulators.serotonin;
    float acetylcholine = neuron.neuromodulators.acetylcholine;
    float noradrenaline = neuron.neuromodulators.noradrenaline;
    
    // 1. Modulate intrinsic excitability
    
    // Dopamine: Increases excitability of neurons in the "direct pathway"
    //           Decreases excitability of neurons in the "indirect pathway"
    float dopamine_excitability = 0.0f;
    if (neuron.neuron_type == NEURON_DIRECT_PATHWAY) {
        dopamine_excitability = dopamine * DOPAMINE_DIRECT_FACTOR;
    } else if (neuron.neuron_type == NEURON_INDIRECT_PATHWAY) {
        dopamine_excitability = -dopamine * DOPAMINE_INDIRECT_FACTOR;
    }
    
    // Serotonin: Generally decreases excitability, promoting risk aversion
    float serotonin_excitability = -serotonin * SEROTONIN_EXCITABILITY_FACTOR;
    
    // Acetylcholine: Increases excitability, promoting attention and learning
    float acetylcholine_excitability = acetylcholine * ACETYLCHOLINE_EXCITABILITY_FACTOR;
    
    // Noradrenaline: Increases signal-to-noise ratio
    // - Increases excitability of highly active neurons
    // - Decreases excitability of weakly active neurons
    float activity_level = neuron.activity_level;
    float noradrenaline_excitability = 0.0f;
    if (activity_level > 0.5f) {
        noradrenaline_excitability = noradrenaline * NORADRENALINE_EXCITABILITY_FACTOR;
    } else {
        noradrenaline_excitability = -noradrenaline * NORADRENALINE_EXCITABILITY_FACTOR * 0.5f;
    }
    
    // Combine all effects
    float total_excitability_modulation = dopamine_excitability + 
                                         serotonin_excitability + 
                                         acetylcholine_excitability + 
                                         noradrenaline_excitability;
    
    // Apply with adaptation
    neuron.excitability_modulation = neuron.excitability_modulation * (1.0f - neuron.adaptation_rate * dt) + 
                                    total_excitability_modulation * neuron.adaptation_rate * dt;
    
    // 2. Modulate ion channel properties
    
    // Dopamine: Enhances NMDA currents
    if (dopamine > 0.1f) {
        // Increase NMDA conductance
        for (int c = 0; c < neuron.compartment_count; c++) {
            neuron.channels.nmda_g[c] *= (1.0f + dopamine * DOPAMINE_NMDA_FACTOR * dt);
        }
    }
    
    // Acetylcholine: Reduces K+ currents, increasing excitability
    if (acetylcholine > 0.1f) {
        // Reduce potassium conductance
        for (int c = 0; c < neuron.compartment_count; c++) {
            // Modify Hodgkin-Huxley K+ conductance
            if (c == 0) {  // Soma
                neuron.k_conductance_modulation = 1.0f - acetylcholine * ACETYLCHOLINE_K_FACTOR;
            } else {  // Dendrites
                // Apply to dendritic compartments as well
                neuron.k_conductance_modulation_dendrites[c] = 1.0f - acetylcholine * ACETYLCHOLINE_K_FACTOR;
            }
        }
    }
    
    // Noradrenaline: Enhances HCN channel activity
    if (noradrenaline > 0.1f) {
        // Increase HCN conductance
        for (int c = 0; c < neuron.compartment_count; c++) {
            neuron.channels.hcn_h[c] *= (1.0f + noradrenaline * NORADRENALINE_HCN_FACTOR * dt);
        }
    }
    
    // 3. Modulate calcium dynamics
    
    // Dopamine: Enhances calcium influx
    if (dopamine > 0.1f) {
        for (int c = 0; c < neuron.compartment_count; c++) {
            neuron.ca_influx_modulation[c] = 1.0f + dopamine * DOPAMINE_CA_FACTOR;
        }
    }
    
    // 4. Update neuron state based on modulations
    
    // Apply excitability modulation to resting potential
    neuron.resting_potential_modulated = neuron.resting_potential + neuron.excitability_modulation;
    
    // Apply excitability modulation to spike threshold
    neuron.spike_threshold_modulated = neuron.spike_threshold - neuron.excitability_modulation * 0.5f;
}
```

#### 3.2 Create Synapse Modulation Kernel

```cpp
__global__ void modulateSynapsesKernel(GPUSynapse* synapses, GPUNeuronState* neurons, 
                                      int num_synapses, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Get neuromodulator levels from post-synaptic neuron
    float dopamine = neurons[post_idx].neuromodulators.dopamine;
    float serotonin = neurons[post_idx].neuromodulators.serotonin;
    float acetylcholine = neurons[post_idx].neuromodulators.acetylcholine;
    float noradrenaline = neurons[post_idx].neuromodulators.noradrenaline;
    
    // 1. Modulate synaptic plasticity
    
    // Dopamine: Enhances plasticity for reward-related learning
    float dopamine_plasticity = dopamine * DOPAMINE_PLASTICITY_FACTOR;
    
    // Acetylcholine: Enhances plasticity for attention-related learning
    float acetylcholine_plasticity = acetylcholine * ACETYLCHOLINE_PLASTICITY_FACTOR;
    
    // Noradrenaline: Enhances plasticity during high arousal/stress
    float noradrenaline_plasticity = noradrenaline * NORADRENALINE_PLASTICITY_FACTOR;
    
    // Combine effects
    float plasticity_modulation = dopamine_plasticity + acetylcholine_plasticity + noradrenaline_plasticity;
    
    // Update synapse plasticity rate
    synapse.plasticity_rate = BASE_PLASTICITY_RATE * (1.0f + plasticity_modulation);
    
    // 2. Modulate synaptic transmission
    
    // Dopamine: Enhances excitatory transmission in direct pathway
    float dopamine_transmission = 0.0f;
    if (neurons[post_idx].neuron_type == NEURON_DIRECT_PATHWAY && synapse.weight > 0) {
        dopamine_transmission = dopamine * DOPAMINE_TRANSMISSION_FACTOR;
    }
    
    // Serotonin: Enhances inhibitory transmission
    float serotonin_transmission = 0.0f;
    if (synapse.weight < 0) {
        serotonin_transmission = serotonin * SEROTONIN_INHIBITORY_FACTOR;
    }
    
    // Acetylcholine: Enhances transmission in attention-related pathways
    float acetylcholine_transmission = 0.0f;
    if (neurons[post_idx].neuron_type == NEURON_ATTENTION) {
        acetylcholine_transmission = acetylcholine * ACETYLCHOLINE_TRANSMISSION_FACTOR;
    }
    
    // Combine effects
    float transmission_modulation = dopamine_transmission + serotonin_transmission + acetylcholine_transmission;
    
    // Update effective synaptic weight
    synapse.effective_weight = synapse.weight * (1.0f + transmission_modulation);
    
    // 3. Modulate eligibility traces
    
    // Dopamine: Enhances eligibility trace formation
    if (dopamine > 0.1f) {
        synapse.fast_trace *= (1.0f + dopamine * DOPAMINE_ELIGIBILITY_FACTOR * dt);
    }
    
    // Acetylcholine: Enhances tag formation
    if (acetylcholine > 0.1f && fabsf(synapse.medium_trace) > TAG_THRESHOLD * 0.5f) {
        float tag_enhancement = acetylcholine * ACETYLCHOLINE_TAG_FACTOR * dt;
        synapse.tag_strength += (synapse.medium_trace > 0) ? tag_enhancement : -tag_enhancement;
    }
}
```

#### 3.3 Implement Network State Regulation Kernel

```cpp
__global__ void regulateNetworkStateKernel(GPUNeuronState* neurons, 
                                         GlobalNeuromodulators* global_nm,
                                         float* network_state_params,
                                         int N) {
    // This kernel computes global network state parameters based on neuromodulator levels
    // Only one thread should execute this
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Get global neuromodulator levels
    float dopamine = global_nm->levels[DOPAMINE];
    float serotonin = global_nm->levels[SEROTONIN];
    float acetylcholine = global_nm->levels[ACETYLCHOLINE];
    float noradrenaline = global_nm->levels[NORADRENALINE];
    
    // Compute network state parameters
    
    // 1. Exploration vs. exploitation balance
    // High dopamine and noradrenaline promote exploration
    // High serotonin promotes exploitation
    float exploration_factor = (dopamine * DOPAMINE_EXPLORATION_FACTOR + 
                              noradrenaline * NORADRENALINE_EXPLORATION_FACTOR) - 
                              (serotonin * SEROTONIN_EXPLOITATION_FACTOR);
    
    // 2. Risk sensitivity
    // High serotonin decreases risk-taking
    // High dopamine increases risk-taking
    float risk_sensitivity = (dopamine * DOPAMINE_RISK_FACTOR) - 
                            (serotonin * SEROTONIN_RISK_FACTOR);
    
    // 3. Learning rate
    // High acetylcholine and dopamine increase learning rate
    float learning_rate_modulation = (acetylcholine * ACETYLCHOLINE_LEARNING_FACTOR + 
                                    dopamine * DOPAMINE_LEARNING_FACTOR);
    
    // 4. Attention focus
    // High acetylcholine increases attention
    float attention_focus = acetylcholine * ACETYLCHOLINE_ATTENTION_FACTOR;
    
    // Store computed parameters
    network_state_params[0] = exploration_factor;
    network_state_params[1] = risk_sensitivity;
    network_state_params[2] = learning_rate_modulation;
    network_state_params[3] = attention_focus;
}
```

### 4. Integrate with Market Volatility Adaptation

#### 4.1 Create Market State Processor

```cpp
// MarketStateProcessor.h
#ifndef MARKET_STATE_PROCESSOR_H
#define MARKET_STATE_PROCESSOR_H

#include "NeuromodulatorSystem.h"

class MarketStateProcessor {
private:
    // Market state tracking
    float current_volatility;
    float volatility_history[VOLATILITY_HISTORY_LENGTH];
    int volatility_history_index;
    
    float current_trend;
    float trend_history[TREND_HISTORY_LENGTH];
    int trend_history_index;
    
    float current_pnl;
    float pnl_history[PNL_HISTORY_LENGTH];
    int pnl_history_index;
    
    // Derived metrics
    float volatility_change_rate;
    float trend_reversal_frequency;
    float pnl_stability;
    
    // Internal methods
    void updateVolatilityMetrics();
    void updateTrendMetrics();
    void updatePnLMetrics();
    
public:
    MarketStateProcessor();
    
    // Update methods
    void processMarketUpdate(float price, float volume, float spread);
    void processTrade(float entry_price, float exit_price, float volume, bool is_long);
    
    // Neuromodulator production
    void computeNeuromodulatorProduction(GlobalNeuromodulators* neuromodulators);
    
    // Getters
    float getVolatility() const { return current_volatility; }
    float getTrend() const { return current_trend; }
    float getPnL() const { return current_pnl; }
    float getVolatilityChangeRate() const { return volatility_change_rate; }
    float getTrendReversalFrequency() const { return trend_reversal_frequency; }
    float getPnLStability() const { return pnl_stability; }
};

#endif // MARKET_STATE_PROCESSOR_H
```

#### 4.2 Implement Market-to-Neuromodulator Mapping

```cpp
void MarketStateProcessor::computeNeuromodulatorProduction(GlobalNeuromodulators* neuromodulators) {
    // This method maps market state to neuromodulator production rates
    
    // 1. Dopamine production
    // - Increased by positive PnL (reward)
    // - Decreased by negative PnL (punishment)
    float dopamine_production = current_pnl * PNL_TO_DOPAMINE_FACTOR;
    
    // Add baseline production
    dopamine_production += DOPAMINE_BASELINE_PRODUCTION;
    
    // 2. Serotonin production
    // - Increased during stable, predictable markets
    // - Decreased during volatile, unpredictable markets
    float serotonin_production = SEROTONIN_BASELINE_PRODUCTION - 
                               (current_volatility * VOLATILITY_TO_SEROTONIN_FACTOR) +
                               (pnl_stability * STABILITY_TO_SEROTONIN_FACTOR);
    
    // 3. Acetylcholine production
    // - Increased during trend changes (requiring attention)
    // - Increased during high information content in market
    float acetylcholine_production = ACETYLCHOLINE_BASELINE_PRODUCTION +
                                   (trend_reversal_frequency * TREND_CHANGE_TO_ACETYLCHOLINE_FACTOR);
    
    // 4. Noradrenaline production
    // - Increased during high volatility (stress response)
    // - Increased during rapid volatility changes (surprise)
    float noradrenaline_production = NORADRENALINE_BASELINE_PRODUCTION +
                                   (current_volatility * VOLATILITY_TO_NORADRENALINE_FACTOR) +
                                   (volatility_change_rate * VOLATILITY_CHANGE_TO_NORADRENALINE_FACTOR);
    
    // Update production rates in global neuromo
