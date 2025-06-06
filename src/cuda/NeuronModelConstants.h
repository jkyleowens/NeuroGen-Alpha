// NeuronModelConstants.h
// Constants for the biologically realistic neuron model
#ifndef NEURON_MODEL_CONSTANTS_H
#define NEURON_MODEL_CONSTANTS_H

// Compartment types
#define COMPARTMENT_INACTIVE 0
#define COMPARTMENT_SOMA 1
#define COMPARTMENT_BASAL 2
#define COMPARTMENT_APICAL 3
#define COMPARTMENT_AXON 4

// Receptor types
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3

// Neuron types
#define NEURON_REGULAR 0
#define NEURON_DIRECT_PATHWAY 1
#define NEURON_INDIRECT_PATHWAY 2
#define NEURON_REWARD_PREDICTION 3
#define NEURON_ATTENTION 4

// Hodgkin-Huxley parameters
#define HH_G_NA 120.0f  // Sodium conductance
#define HH_G_K 36.0f    // Potassium conductance
#define HH_G_L 0.3f     // Leak conductance
#define HH_E_NA 50.0f   // Sodium reversal potential
#define HH_E_K -77.0f   // Potassium reversal potential
#define HH_E_L -54.387f // Leak reversal potential

// Biophysical constants
#define RESTING_POTENTIAL -65.0f
#define SPIKE_THRESHOLD -40.0f
#define RESTING_CA_CONCENTRATION 0.0001f
#define DENDRITIC_SPIKE_THRESHOLD -30.0f
#define DENDRITIC_SPIKE_DURATION 5.0f
#define DENDRITIC_SPIKE_CA_INFLUX 0.1f
#define DENDRITIC_SPIKE_PROPAGATION_STRENGTH 5.0f
#define NMDA_THRESHOLD 0.5f
#define NMDA_CA_INFLUX 0.05f
#define NMDA_CA_FRACTION 0.1f

// Coupling conductances
#define BASAL_COUPLING_CONDUCTANCE 0.5f
#define APICAL_COUPLING_CONDUCTANCE 0.3f

// Synaptic scaling factors
#define BASAL_DENDRITE_SCALING 1.2f
#define APICAL_DENDRITE_SCALING 0.8f

// Calcium dynamics
#define CA_BUFFER_CAPACITY 10.0f
#define CA_BUFFER_KD 0.001f
#define CA_EXTRUSION_RATE 0.1f
#define CA_DIFFUSION_RATE 0.1f

// AMPA receptor parameters
#define AMPA_TAU_RISE 0.5f
#define AMPA_TAU_DECAY 3.0f
#define AMPA_REVERSAL 0.0f

// NMDA receptor parameters
#define NMDA_TAU_RISE 5.0f
#define NMDA_TAU_DECAY 50.0f
#define NMDA_REVERSAL 0.0f
#define NMDA_MG_CONC 1.0f

// GABA-A receptor parameters
#define GABA_A_TAU_RISE 1.0f
#define GABA_A_TAU_DECAY 7.0f
#define GABA_A_REVERSAL -70.0f

// GABA-B receptor parameters
#define GABA_B_TAU_RISE 50.0f
#define GABA_B_TAU_DECAY 100.0f
#define GABA_B_TAU_K 10.0f
#define GABA_B_REVERSAL -90.0f

// STDP parameters
#define STDP_TAU_PLUS 20.0f
#define STDP_TAU_MINUS 20.0f
#define STDP_A_PLUS_EXC 0.005f
#define STDP_A_MINUS_EXC 0.0025f
#define STDP_A_PLUS_INH -0.001f
#define STDP_A_MINUS_INH 0.002f

// Eligibility trace parameters
#define FAST_TRACE_TAU 50.0f
#define MEDIUM_TRACE_TAU 5000.0f
#define SLOW_TRACE_TAU 100000.0f
#define TAG_TAU 30000.0f
#define MAX_ELIGIBILITY_TRACE 5.0f
#define MAX_TAG_STRENGTH 1.0f
#define TAG_THRESHOLD 0.5f
#define TAG_CREATION_RATE 0.1f
#define FAST_TO_MEDIUM_TRANSFER 0.1f
#define MEDIUM_TO_SLOW_TRANSFER 0.05f

// Reward modulation parameters
#define BASELINE_DOPAMINE 0.0f
#define FAST_TRACE_MODULATION 1.0f
#define MEDIUM_TRACE_MODULATION 0.5f
#define SLOW_TRACE_MODULATION 0.1f
#define LATE_PHASE_FACTOR 0.2f

// Hebbian learning parameters
#define HEBBIAN_LEARNING_RATE 0.0001f
#define TARGET_ACTIVITY 0.1f
#define MAX_TOTAL_IN_WEIGHT 10.0f
#define MAX_TOTAL_OUT_WEIGHT 20.0f

// Activity tracking
#define ACTIVITY_TAU 1000.0f

// Weight constraints
#define MIN_WEIGHT -2.0f
#define MAX_WEIGHT 2.0f

// Neuromodulator parameters
#define REWARD_TO_DOPAMINE_FACTOR 1.0f
#define DOPAMINE_DIRECT_FACTOR 0.5f
#define DOPAMINE_INDIRECT_FACTOR 0.5f
#define SEROTONIN_EXCITABILITY_FACTOR 0.3f
#define ACETYLCHOLINE_EXCITABILITY_FACTOR 0.4f
#define NORADRENALINE_EXCITABILITY_FACTOR 0.5f
#define DOPAMINE_NMDA_FACTOR 0.2f
#define ACETYLCHOLINE_K_FACTOR 0.3f
#define NORADRENALINE_HCN_FACTOR 0.2f
#define DOPAMINE_CA_FACTOR 0.3f
#define DOPAMINE_PLASTICITY_FACTOR 0.5f
#define ACETYLCHOLINE_PLASTICITY_FACTOR 0.4f
#define NORADRENALINE_PLASTICITY_FACTOR 0.3f
#define DOPAMINE_TRANSMISSION_FACTOR 0.3f
#define SEROTONIN_INHIBITORY_FACTOR 0.4f
#define ACETYLCHOLINE_TRANSMISSION_FACTOR 0.3f
#define DOPAMINE_ELIGIBILITY_FACTOR 0.2f
#define ACETYLCHOLINE_TAG_FACTOR 0.3f
#define DOPAMINE_EXPLORATION_FACTOR 0.5f
#define NORADRENALINE_EXPLORATION_FACTOR 0.4f
#define SEROTONIN_EXPLOITATION_FACTOR 0.5f
#define DOPAMINE_RISK_FACTOR 0.4f
#define SEROTONIN_RISK_FACTOR 0.5f
#define ACETYLCHOLINE_LEARNING_FACTOR 0.4f
#define DOPAMINE_LEARNING_FACTOR 0.5f
#define ACETYLCHOLINE_ATTENTION_FACTOR 0.6f

// Receptor desensitization
#define DESENSITIZATION_RATE 0.01f
#define RESENSITIZATION_RATE 0.001f

// Release parameters
#define RELEASE_DEPLETION_FACTOR 0.3f
#define SUBTHRESHOLD_RELEASE_FACTOR 0.1f

#endif // NEURON_MODEL_CONSTANTS_H
