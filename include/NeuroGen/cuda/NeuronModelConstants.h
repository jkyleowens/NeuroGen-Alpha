// NeuronModelConstants.h
// Constants for the biologically realistic neuron model
#ifndef NEURON_MODEL_CONSTANTS_H
#define NEURON_MODEL_CONSTANTS_H

// --- Physical Properties and Thresholds ---
#define V_REST -65.0f          // Resting membrane potential (mV)
#define V_THRESH -55.0f        // Spike threshold (mV)
#define V_RESET -75.0f         // Reset potential after a spike (mV)

// --- Hodgkin-Huxley Model Parameters ---
#define HH_G_NA 120.0f          // Sodium conductance (mS/cm^2)
#define HH_G_K 36.0f            // Potassium conductance (mS/cm^2)
#define HH_G_L 0.3f             // Leak conductance (mS/cm^2)
#define HH_E_NA 50.0f           // Sodium reversal potential (mV)
#define HH_E_K -77.0f           // Potassium reversal potential (mV)
#define HH_E_L -54.387f         // Leak reversal potential (mV)
#define MEMBRANE_CAPACITANCE 1.0f // uF/cm^2

// --- Dendritic Spike Parameters ---
#define DENDRITIC_SPIKE_THRESHOLD -40.0f
#define NMDA_CA_FRACTION 0.1f

// --- Calcium Dynamics ---
#define RESTING_CA_CONC 1.0e-4f // Resting calcium concentration (mM) - (100 nM)
#define CA_DECAY_TAU 200.0f     // Calcium decay time constant (ms)
#define CA_SPIKE_INCREMENT 5.0e-4f // Calcium increase per spike (mM)

// --- Spike Frequency Adaptation ---
#define ADAPTATION_INCREMENT 0.05f  // Amount adaptation current increases per spike
#define ADAPTATION_TAU 250.0f       // Time constant for adaptation current decay (ms)

#endif // NEURON_MODEL_CONSTANTS_H