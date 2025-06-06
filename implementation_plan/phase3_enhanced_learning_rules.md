# Phase 3: Enhanced Learning Rules

## Overview

The current STDP implementation is simplistic and doesn't capture the complexity of biological learning. This phase will implement multi-factor STDP with eligibility traces and reward modulation, enabling more sophisticated learning and adaptation.

## Current Implementation Analysis

From the code analysis, we found:

- Basic STDP is implemented but lacks compartment and synapse-type specificity
- Eligibility traces are not fully implemented
- Reward modulation is simplistic and doesn't account for temporal credit assignment
- There's no differentiation between learning rules for different synapse types

## Implementation Tasks

### 1. Implement Multi-Factor STDP

#### 1.1 Create Enhanced STDP Kernel

```cpp
__global__ void enhancedSTDPKernel(GPUSynapse* synapses, GPUNeuronState* neurons, 
                                  float current_time, float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    int compartment = synapse.post_compartment;
    int receptor = synapse.receptor_index;
    
    // Get spike times
    float t_pre = synapse.last_pre_spike_time;
    float t_post = neurons[post_idx].last_spike_time;
    
    // Skip if no recent spikes
    if (t_pre < 0.0f || t_post < 0.0f) return;
    
    // Calculate spike timing difference
    float dt_spike = t_post - t_pre;
    
    // Determine synapse type (excitatory or inhibitory)
    bool is_excitatory = synapse.weight > 0.0f;
    
    // Determine compartment type
    int comp_type = neurons[post_idx].compartment_types[compartment];
    
    // Base STDP parameters
    float A_plus = 0.0f;  // Potentiation amplitude
    float A_minus = 0.0f; // Depression amplitude
    float tau_plus = 20.0f;  // Potentiation time constant (ms)
    float tau_minus = 20.0f; // Depression time constant (ms)
    
    // Adjust STDP parameters based on synapse and compartment type
    if (is_excitatory) {
        if (comp_type == COMPARTMENT_BASAL) {
            // Basal excitatory synapses: standard STDP
            A_plus = 0.005f;
            A_minus = 0.0025f;
        } else if (comp_type == COMPARTMENT_APICAL) {
            // Apical excitatory synapses: stronger potentiation
            A_plus = 0.008f;
            A_minus = 0.002f;
            tau_plus = 25.0f; // Longer time window
        } else {
            // Somatic excitatory synapses: balanced
            A_plus = 0.004f;
            A_minus = 0.004f;
        }
    } else {
        // Inhibitory synapses: anti-Hebbian STDP
        if (comp_type == COMPARTMENT_BASAL || comp_type == COMPARTMENT_APICAL) {
            A_plus = -0.001f;  // Depression for pre-post
            A_minus = 0.002f;  // Potentiation for post-pre
        } else {
            // Somatic inhibitory synapses: stronger effect
            A_plus = -0.002f;
            A_minus = 0.004f;
        }
    }
    
    // Calculate STDP weight change
    float dw = 0.0f;
    
    if (dt_spike > 0.0f) {
        // Post-synaptic spike after pre-synaptic spike (potentiation)
        dw = A_plus * expf(-dt_spike / tau_plus);
    } else if (dt_spike < 0.0f) {
        // Post-synaptic spike before pre-synaptic spike (depression)
        dw = -A_minus * expf(dt_spike / tau_minus);
    }
    
    // Metaplasticity: adjust learning rate based on recent activity
    float activity_factor = 1.0f;
    if (synapse.activity_metric > 0.5f) {
        // High recent activity: reduce learning rate to prevent runaway potentiation
        activity_factor = 0.5f;
    } else if (synapse.activity_metric < 0.1f) {
        // Low recent activity: boost learning rate to encourage exploration
        activity_factor = 1.5f;
    }
    
    dw *= activity_factor;
    
    // Update eligibility trace instead of directly changing weight
    // Eligibility trace decays over time and is modulated by reward signals
    synapse.eligibility_trace += dw;
    
    // Apply decay to eligibility trace
    synapse.eligibility_trace *= (1.0f - dt / ELIGIBILITY_TRACE_TAU);
    
    // Ensure eligibility trace stays in reasonable range
    if (synapse.eligibility_trace > MAX_ELIGIBILITY_TRACE) {
        synapse.eligibility_trace = MAX_ELIGIBILITY_TRACE;
    } else if (synapse.eligibility_trace < -MAX_ELIGIBILITY_TRACE) {
        synapse.eligibility_trace = -MAX_ELIGIBILITY_TRACE;
    }
}
```

#### 1.2 Implement Receptor-Specific STDP Rules

Create specialized STDP rules for different receptor types:

```cpp
__device__ float calculateNMDAWeightChange(float dt_spike, float ca_conc) {
    // NMDA receptors have calcium-dependent plasticity
    // Implement BCM-like rule where moderate calcium leads to depression,
    // high calcium leads to potentiation
    
    float theta = 0.0005f; // Threshold between depression and potentiation
    
    if (ca_conc < theta) {
        // Low calcium: minimal change
        return 0.0f;
    } else if (ca_conc < 2.0f * theta) {
        // Moderate calcium: depression
        return -0.005f * (ca_conc / theta);
    } else {
        // High calcium: potentiation
        return 0.01f * ((ca_conc - 2.0f * theta) / theta);
    }
}

__device__ float calculateAMPAWeightChange(float dt_spike) {
    // AMPA receptors follow standard STDP
    float A_plus = 0.005f;
    float A_minus = 0.0025f;
    float tau_plus = 20.0f;
    float tau_minus = 20.0f;
    
    if (dt_spike > 0.0f) {
        return A_plus * expf(-dt_spike / tau_plus);
    } else if (dt_spike < 0.0f) {
        return -A_minus * expf(dt_spike / tau_minus);
    }
    
    return 0.0f;
}

__device__ float calculateGABAWeightChange(float dt_spike, bool is_gaba_a) {
    // GABA receptors have different plasticity rules
    float A_plus, A_minus, tau_plus, tau_minus;
    
    if (is_gaba_a) {
        // GABA-A (fast inhibition)
        A_plus = -0.001f;  // Depression for pre-post
        A_minus = 0.002f;  // Potentiation for post-pre
        tau_plus = 20.0f;
        tau_minus = 20.0f;
    } else {
        // GABA-B (slow inhibition)
        A_plus = -0.0005f;  // Weaker effect for slow inhibition
        A_minus = 0.001f;
        tau_plus = 50.0f;   // Longer time window
        tau_minus = 50.0f;
    }
    
    if (dt_spike > 0.0f) {
        return A_plus * expf(-dt_spike / tau_plus);
    } else if (dt_spike < 0.0f) {
        return -A_minus * expf(dt_spike / tau_minus);
    }
    
    return 0.0f;
}
```

### 2. Implement Multi-Timescale Eligibility Traces

#### 2.1 Update Synapse Structure

```cpp
struct GPUSynapse {
    // Existing fields
    // ...
    
    // Enhanced eligibility trace system
    float fast_trace;        // Fast eligibility trace (tens of ms)
    float medium_trace;      // Medium eligibility trace (seconds)
    float slow_trace;        // Slow eligibility trace (minutes)
    float tag_strength;      // Synaptic tag for late-phase plasticity
    
    // Metaplasticity variables
    float meta_weight;       // Metaplastic weight (controls plasticity threshold)
    float recent_activity;   // Measure of recent activity
};
```

#### 2.2 Create Eligibility Trace Kernel

```cpp
__global__ void eligibilityTraceKernel(GPUSynapse* synapses, float dt, float current_time, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    // Decay rates for different timescales
    float fast_decay = expf(-dt / FAST_TRACE_TAU);
    float medium_decay = expf(-dt / MEDIUM_TRACE_TAU);
    float slow_decay = expf(-dt / SLOW_TRACE_TAU);
    float tag_decay = expf(-dt / TAG_TAU);
    
    // Decay traces
    synapse.fast_trace *= fast_decay;
    synapse.medium_trace *= medium_decay;
    synapse.slow_trace *= slow_decay;
    synapse.tag_strength *= tag_decay;
    
    // Cascade from fast to medium to slow traces
    // This implements the synaptic tagging and capture mechanism
    synapse.medium_trace += synapse.fast_trace * FAST_TO_MEDIUM_TRANSFER * dt;
    synapse.slow_trace += synapse.medium_trace * MEDIUM_TO_SLOW_TRANSFER * dt;
    
    // Update synaptic tag based on medium trace
    // Tags are created when medium trace exceeds a threshold
    if (fabsf(synapse.medium_trace) > TAG_THRESHOLD) {
        float tag_increment = (synapse.medium_trace > 0) ? 
                             TAG_CREATION_RATE * dt : -TAG_CREATION_RATE * dt;
        synapse.tag_strength += tag_increment;
        
        // Clamp tag strength
        if (synapse.tag_strength > MAX_TAG_STRENGTH) {
            synapse.tag_strength = MAX_TAG_STRENGTH;
        } else if (synapse.tag_strength < -MAX_TAG_STRENGTH) {
            synapse.tag_strength = -MAX_TAG_STRENGTH;
        }
    }
    
    // Decay recent activity metric
    synapse.recent_activity *= expf(-dt / ACTIVITY_TAU);
}
```

### 3. Implement Reward Modulation

#### 3.1 Create Reward Modulation Kernel

```cpp
__global__ void rewardModulationKernel(GPUSynapse* synapses, float reward_signal, 
                                      float dopamine_level, float current_time, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    // Compute effective reward signal
    // This combines the external reward with the internal dopamine level
    float effective_reward = reward_signal + dopamine_level - BASELINE_DOPAMINE;
    
    // Skip if no significant reward signal
    if (fabsf(effective_reward) < 0.01f) return;
    
    // Apply reward modulation to weight based on eligibility traces
    // Fast trace: immediate reinforcement
    // Medium trace: delayed reinforcement
    // Slow trace: long-term memory consolidation
    
    float dw_fast = synapse.fast_trace * effective_reward * FAST_TRACE_MODULATION;
    float dw_medium = synapse.medium_trace * effective_reward * MEDIUM_TRACE_MODULATION;
    float dw_slow = synapse.slow_trace * effective_reward * SLOW_TRACE_MODULATION;
    
    // Combine weight changes
    float dw_total = dw_fast + dw_medium + dw_slow;
    
    // Apply learning rate and plasticity modulation
    dw_total *= synapse.plasticity_rate;
    
    // Update weight
    synapse.weight += dw_total;
    
    // Apply weight constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
    
    // Late-phase plasticity: if there's a strong tag and significant reward,
    // trigger protein synthesis-dependent long-term plasticity
    if (fabsf(synapse.tag_strength) > 0.5f && fabsf(effective_reward) > 0.5f) {
        // Sign of tag and reward must match for reinforcement
        if (signbit(synapse.tag_strength) == signbit(effective_reward)) {
            // Trigger late-phase plasticity (protein synthesis)
            float late_dw = synapse.tag_strength * effective_reward * LATE_PHASE_FACTOR;
            
            // Late-phase changes directly affect the weight, bypassing eligibility traces
            synapse.weight += late_dw;
            
            // Apply weight constraints again
            if (synapse.weight > MAX_WEIGHT) {
                synapse.weight = MAX_WEIGHT;
            } else if (synapse.weight < MIN_WEIGHT) {
                synapse.weight = MIN_WEIGHT;
            }
            
            // Consume the tag
            synapse.tag_strength *= 0.5f;
        }
    }
}
```

#### 3.2 Implement Prediction Error Computation

```cpp
__global__ void predictionErrorKernel(GPUNeuronState* neurons, float actual_reward, 
                                     float* predicted_reward, float* prediction_error, int N) {
    // This kernel computes the reward prediction error (RPE)
    // RPE = actual_reward - predicted_reward
    // This is used to modulate dopamine release
    
    // Identify neurons that predict reward
    float total_prediction = 0.0f;
    int prediction_count = 0;
    
    for (int i = 0; i < N; i++) {
        if (neurons[i].neuron_type == NEURON_REWARD_PREDICTION && neurons[i].spiked) {
            total_prediction += neurons[i].activity_level;
            prediction_count++;
        }
    }
    
    // Compute average prediction if any prediction neurons fired
    if (prediction_count > 0) {
        *predicted_reward = total_prediction / prediction_count;
    } else {
        *predicted_reward = 0.0f;
    }
    
    // Compute prediction error
    *prediction_error = actual_reward - *predicted_reward;
}
```

### 4. Implement Hebbian Learning

#### 4.1 Create Hebbian Learning Kernel

```cpp
__global__ void hebbianLearningKernel(GPUSynapse* synapses, GPUNeuronState* neurons, 
                                     float dt, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Skip inactive synapses
    if (synapse.active == 0) return;
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    // Get neuron activity levels (rate-based)
    float pre_activity = neurons[pre_idx].activity_level;
    float post_activity = neurons[post_idx].activity_level;
    
    // Skip if activity is too low
    if (pre_activity < 0.01f || post_activity < 0.01f) return;
    
    // Basic Hebbian rule: weight change proportional to pre * post activity
    float dw = HEBBIAN_LEARNING_RATE * pre_activity * post_activity * dt;
    
    // Apply homeostatic scaling based on post-synaptic activity
    // If post-synaptic neuron is too active, scale down incoming weights
    float homeostatic_factor = 1.0f;
    if (post_activity > TARGET_ACTIVITY) {
        homeostatic_factor = TARGET_ACTIVITY / post_activity;
    }
    
    dw *= homeostatic_factor;
    
    // Update weight
    synapse.weight += dw;
    
    // Apply weight constraints
    if (synapse.weight > MAX_WEIGHT) {
        synapse.weight = MAX_WEIGHT;
    } else if (synapse.weight < MIN_WEIGHT) {
        synapse.weight = MIN_WEIGHT;
    }
}
```

#### 4.2 Implement Synaptic Normalization

```cpp
__global__ void synapticNormalizationKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip inactive neurons
    if (neuron.active == 0) return;
    
    // Compute total incoming and outgoing synaptic weights
    float total_in_weight = 0.0f;
    float total_out_weight = 0.0f;
    int in_count = 0;
    int out_count = 0;
    
    // Count incoming synapses (this is inefficient but illustrative)
    for (int s = 0; s < num_synapses; s++) {
        if (synapses[s].active == 0) continue;
        
        if (synapses[s].post_neuron_idx == idx && synapses[s].weight > 0) {
            total_in_weight += synapses[s].weight;
            in_count++;
        }
        
        if (synapses[s].pre_neuron_idx == idx && synapses[s].weight > 0) {
            total_out_weight += synapses[s].weight;
            out_count++;
        }
    }
    
    // Normalize incoming weights if needed
    if (in_count > 0 && total_in_weight > MAX_TOTAL_IN_WEIGHT) {
        float scale_factor = MAX_TOTAL_IN_WEIGHT / total_in_weight;
        
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].post_neuron_idx == idx && synapses[s].weight > 0) {
                synapses[s].weight *= scale_factor;
            }
        }
    }
    
    // Normalize outgoing weights if needed
    if (out_count > 0 && total_out_weight > MAX_TOTAL_OUT_WEIGHT) {
        float scale_factor = MAX_TOTAL_OUT_WEIGHT / total_out_weight;
        
        for (int s = 0; s < num_synapses; s++) {
            if (synapses[s].active == 0) continue;
            
            if (synapses[s].pre_neuron_idx == idx && synapses[s].weight > 0) {
                synapses[s].weight *= scale_factor;
            }
        }
    }
}
```

### 5. Update Constants and Definitions

Create a new header file with learning rule constants:

```cpp
// LearningRuleConstants.h
#ifndef LEARNING_RULE_CONSTANTS_H
#define LEARNING_RULE_CONSTANTS_H

// STDP parameters
#define STDP_TAU_PLUS 20.0f          // Potentiation time constant (ms)
#define STDP_TAU_MINUS 20.0f         // Depression time constant (ms)
#define STDP_A_PLUS_EXC 0.005f       // Potentiation amplitude for excitatory synapses
#define STDP_A_MINUS_EXC 0.0025f     // Depression amplitude for excitatory synapses
#define STDP_A_PLUS_INH -0.001f      // Potentiation amplitude for inhibitory synapses
#define STDP_A_MINUS_INH 0.002f      // Depression amplitude for inhibitory synapses

// Eligibility trace parameters
#define FAST_TRACE_TAU 50.0f         // Fast trace decay time constant (ms)
#define MEDIUM_TRACE_TAU 5000.0f     // Medium trace decay time constant (ms)
#define SLOW_TRACE_TAU 100000.0f     // Slow trace decay time constant (ms)
#define TAG_TAU 30000.0f             // Synaptic tag decay time constant (ms)
#define MAX_ELIGIBILITY_TRACE 5.0f   // Maximum eligibility trace value
#define MAX_TAG_STRENGTH 1.0f        // Maximum synaptic tag strength
#define TAG_THRESHOLD 0.5f           // Threshold for tag creation
#define TAG_CREATION_RATE 0.1f       // Rate of tag creation
#define FAST_TO_MEDIUM_TRANSFER 0.1f // Transfer rate from fast to medium trace
#define MEDIUM_TO_SLOW_TRANSFER 0.05f // Transfer rate from medium to slow trace

// Reward modulation parameters
#define BASELINE_DOPAMINE 0.0f       // Baseline dopamine level
#define FAST_TRACE_MODULATION 1.0f   // Modulation factor for fast trace
#define MEDIUM_TRACE_MODULATION 0.5f // Modulation factor for medium trace
#define SLOW_TRACE_MODULATION 0.1f   // Modulation factor for slow trace
#define LATE_PHASE_FACTOR 0.2f       // Factor for late-phase plasticity

// Hebbian learning parameters
#define HEBBIAN_LEARNING_RATE 0.0001f // Basic Hebbian learning rate
#define TARGET_ACTIVITY 0.1f         // Target activity level for homeostasis
#define MAX_TOTAL_IN_WEIGHT 10.0f    // Maximum total incoming weight
#define MAX_TOTAL_OUT_WEIGHT 20.0f   // Maximum total outgoing weight

// Activity tracking
#define ACTIVITY_TAU 1000.0f         // Activity metric decay time constant (ms)

// Weight constraints
#define MIN_WEIGHT -2.0f             // Minimum synaptic weight
#define MAX_WEIGHT 2.0f              // Maximum synaptic weight

#endif // LEARNING_RULE_CONSTANTS_H
```

## Integration Plan

1. Create the new header files with constants and structures
2. Implement the enhanced STDP kernel with compartment and receptor-specific rules
3. Add the multi-timescale eligibility trace system
4. Implement the reward modulation kernel
5. Add the Hebbian learning and synaptic normalization kernels
6. Update the network simulation loop to call these kernels at appropriate times
7. Test the learning rules with simple scenarios to ensure proper functioning

## Expected Outcomes

- More biologically realistic learning rules
- Improved temporal credit assignment through eligibility traces
- Better adaptation to reward signals
- More stable network dynamics through homeostatic mechanisms
- Foundation for implementing neuromodulation in Phase 4

## Dependencies

- Phase 3 depends on the enhanced neuron model from Phase 1
- Phase 3 depends on the ion channel dynamics from Phase 2
- The neuromodulation system in Phase 4 will build upon the learning rules from Phase 3
