#ifndef ION_CHANNEL_CONSTANTS_H
#define ION_CHANNEL_CONSTANTS_H

// Receptor type indices (consistent with Phase 1)
#define RECEPTOR_AMPA       0
#define RECEPTOR_NMDA       1
#define RECEPTOR_GABA_A     2
#define RECEPTOR_GABA_B     3
#define NUM_RECEPTOR_TYPES  4

// Voltage-gated channel indices
#define CHANNEL_CA          0
#define CHANNEL_KCA         1
#define CHANNEL_HCN         2
#define NUM_VG_CHANNELS     3

// ========================================
// CALCIUM DYNAMICS CONSTANTS
// ========================================
#define RESTING_CA_CONCENTRATION    0.0001f   // mM (100 nM)
#define MAX_CA_CONCENTRATION        0.01f     // mM (10 μM)
#define CA_BUFFER_CAPACITY          10.0f     // Relative units
#define CA_BUFFER_KD                0.001f    // mM (1 μM)
#define CA_EXTRUSION_RATE_SOMA      0.2f      // 1/ms (faster in soma)
#define CA_EXTRUSION_RATE_DENDRITE  0.1f      // 1/ms (slower in dendrites)
#define CA_DIFFUSION_RATE           0.05f     // 1/ms (between compartments)

// Volume factors for current-to-concentration conversion
#define CA_VOLUME_FACTOR_SOMA       1.0f      // Larger volume
#define CA_VOLUME_FACTOR_DENDRITE   2.0f      // Smaller volume
#define CA_VOLUME_FACTOR_SPINE      5.0f      // Very small volume

// ========================================
// AMPA RECEPTOR CONSTANTS
// ========================================
#define AMPA_G_MAX_SOMA             1.0f      // nS
#define AMPA_G_MAX_DENDRITE         0.8f      // nS
#define AMPA_TAU_RISE               0.5f      // ms
#define AMPA_TAU_DECAY              3.0f      // ms
#define AMPA_REVERSAL               0.0f      // mV

// ========================================
// NMDA RECEPTOR CONSTANTS
// ========================================
#define NMDA_G_MAX_SOMA             0.5f      // nS
#define NMDA_G_MAX_DENDRITE         1.2f      // nS (higher in dendrites)
#define NMDA_TAU_RISE               5.0f      // ms
#define NMDA_TAU_DECAY              50.0f     // ms
#define NMDA_REVERSAL               0.0f      // mV
#define NMDA_MG_CONC                1.0f      // mM
#define NMDA_CA_FRACTION            0.1f      // Fraction of current carried by Ca2+

// ========================================
// GABA-A RECEPTOR CONSTANTS
// ========================================
#define GABA_A_G_MAX_SOMA           2.0f      // nS
#define GABA_A_G_MAX_DENDRITE       1.5f      // nS
#define GABA_A_TAU_RISE             1.0f      // ms
#define GABA_A_TAU_DECAY            7.0f      // ms
#define GABA_A_REVERSAL             -70.0f    // mV

// ========================================
// GABA-B RECEPTOR CONSTANTS
// ========================================
#define GABA_B_G_MAX_SOMA           1.0f      // nS
#define GABA_B_G_MAX_DENDRITE       0.8f      // nS
#define GABA_B_TAU_RISE             50.0f     // ms
#define GABA_B_TAU_DECAY            100.0f    // ms
#define GABA_B_TAU_K                10.0f     // ms (G-protein activation)
#define GABA_B_REVERSAL             -90.0f    // mV

// ========================================
// VOLTAGE-GATED CALCIUM CHANNEL CONSTANTS
// ========================================
#define CA_G_MAX_SOMA               0.5f      // nS
#define CA_G_MAX_DENDRITE           0.8f      // nS
#define CA_G_MAX_SPINE              0.3f      // nS
#define CA_REVERSAL                 50.0f     // mV
#define CA_V_HALF                   -20.0f    // mV
#define CA_K                        9.0f      // mV
#define CA_TAU_ACT                  1.0f      // ms

// ========================================
// CALCIUM-DEPENDENT POTASSIUM CHANNEL CONSTANTS
// ========================================
#define KCA_G_MAX_SOMA              2.0f      // nS
#define KCA_G_MAX_DENDRITE          1.5f      // nS
#define KCA_REVERSAL                -90.0f    // mV
#define KCA_CA_HALF                 0.0005f   // mM (0.5 μM)
#define KCA_HILL_COEF               2.0f      // Hill coefficient
#define KCA_TAU_ACT                 5.0f      // ms

// ========================================
// HCN CHANNEL CONSTANTS
// ========================================
#define HCN_G_MAX_SOMA              0.2f      // nS
#define HCN_G_MAX_DENDRITE          0.5f      // nS (higher in dendrites)
#define HCN_REVERSAL                -30.0f    // mV
#define HCN_V_HALF                  -80.0f    // mV
#define HCN_K                       -8.0f     // mV (negative for hyperpolarization)
#define HCN_TAU_MIN                 10.0f     // ms
#define HCN_TAU_MAX                 500.0f    // ms
#define HCN_V_TAU                   -80.0f    // mV
#define HCN_K_TAU                   -15.0f    // mV

// ========================================
// COMPARTMENT-SPECIFIC SCALING FACTORS
// ========================================
// These factors adjust channel densities based on compartment type
#define SCALE_FACTOR_SOMA           1.0f
#define SCALE_FACTOR_BASAL          0.8f
#define SCALE_FACTOR_APICAL         1.2f
#define SCALE_FACTOR_SPINE          0.5f

// ========================================
// INTEGRATION AND NUMERICAL CONSTANTS
// ========================================
#define MIN_CONDUCTANCE             1e-9f     // Minimum conductance (nS)
#define MAX_CONDUCTANCE             100.0f    // Maximum conductance (nS)
#define MIN_CA_CONCENTRATION        1e-6f     // Minimum calcium (mM)

// Time constants for numerical stability
#define MIN_TAU                     0.1f      // Minimum time constant (ms)
#define MAX_TAU                     1000.0f   // Maximum time constant (ms)

// ========================================
// COMPARTMENT TYPE DEFINITIONS
// ========================================
// These should match the definitions from Phase 1
#define COMPARTMENT_INACTIVE        0
#define COMPARTMENT_SOMA            1
#define COMPARTMENT_BASAL           2
#define COMPARTMENT_APICAL          3
#define COMPARTMENT_SPINE           4

// ========================================
// CHANNEL PARAMETER INITIALIZATION MACROS
// ========================================
#define INIT_AMPA_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? AMPA_G_MAX_SOMA : AMPA_G_MAX_DENDRITE, \
    .tau_rise = AMPA_TAU_RISE, \
    .tau_decay = AMPA_TAU_DECAY, \
    .reversal = AMPA_REVERSAL \
}

#define INIT_NMDA_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? NMDA_G_MAX_SOMA : NMDA_G_MAX_DENDRITE, \
    .tau_rise = NMDA_TAU_RISE, \
    .tau_decay = NMDA_TAU_DECAY, \
    .reversal = NMDA_REVERSAL, \
    .mg_conc = NMDA_MG_CONC, \
    .ca_fraction = NMDA_CA_FRACTION \
}

#define INIT_GABA_A_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? GABA_A_G_MAX_SOMA : GABA_A_G_MAX_DENDRITE, \
    .tau_rise = GABA_A_TAU_RISE, \
    .tau_decay = GABA_A_TAU_DECAY, \
    .reversal = GABA_A_REVERSAL \
}

#define INIT_GABA_B_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? GABA_B_G_MAX_SOMA : GABA_B_G_MAX_DENDRITE, \
    .tau_rise = GABA_B_TAU_RISE, \
    .tau_decay = GABA_B_TAU_DECAY, \
    .tau_k = GABA_B_TAU_K, \
    .reversal = GABA_B_REVERSAL \
}

#define INIT_CA_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? CA_G_MAX_SOMA : \
             (comp_type == COMPARTMENT_SPINE) ? CA_G_MAX_SPINE : CA_G_MAX_DENDRITE, \
    .reversal = CA_REVERSAL, \
    .v_half = CA_V_HALF, \
    .k = CA_K, \
    .tau_act = CA_TAU_ACT \
}

#define INIT_KCA_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? KCA_G_MAX_SOMA : KCA_G_MAX_DENDRITE, \
    .reversal = KCA_REVERSAL, \
    .ca_half = KCA_CA_HALF, \
    .hill_coef = KCA_HILL_COEF, \
    .tau_act = KCA_TAU_ACT \
}

#define INIT_HCN_CHANNEL(comp_type) { \
    .g_max = (comp_type == COMPARTMENT_SOMA) ? HCN_G_MAX_SOMA : HCN_G_MAX_DENDRITE, \
    .reversal = HCN_REVERSAL, \
    .v_half = HCN_V_HALF, \
    .k = HCN_K, \
    .tau_min = HCN_TAU_MIN, \
    .tau_max = HCN_TAU_MAX, \
    .v_tau = HCN_V_TAU, \
    .k_tau = HCN_K_TAU \
}

#define INIT_CA_DYNAMICS(comp_type) { \
    .resting_ca = RESTING_CA_CONCENTRATION, \
    .buffer_capacity = CA_BUFFER_CAPACITY, \
    .buffer_kd = CA_BUFFER_KD, \
    .extrusion_rate = (comp_type == COMPARTMENT_SOMA) ? CA_EXTRUSION_RATE_SOMA : CA_EXTRUSION_RATE_DENDRITE, \
    .diffusion_rate = CA_DIFFUSION_RATE, \
    .volume_factor = (comp_type == COMPARTMENT_SOMA) ? CA_VOLUME_FACTOR_SOMA : \
                     (comp_type == COMPARTMENT_SPINE) ? CA_VOLUME_FACTOR_SPINE : CA_VOLUME_FACTOR_DENDRITE \
}

#endif // ION_CHANNEL_CONSTANTS_H