// DynamicNeurogenesisFramework.h
#ifndef DYNAMIC_NEUROGENESIS_FRAMEWORK_H
#define DYNAMIC_NEUROGENESIS_FRAMEWORK_H

#include "GPUNeuralStructures.h"
#include "AdvancedReinforcementLearning.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Dynamic neurogenesis framework implementing:
 * - Activity-dependent neural birth and death
 * - Developmental trajectories for new neurons
 * - Migration and integration of new neurons
 * - Experience-dependent specialization
 * - Homeostatic regulation of neuron numbers
 */

// ========================================
// NEUROGENESIS CONSTANTS
// ========================================

// Neurogenesis control parameters
#define NEUROGENESIS_BASE_RATE      0.001f    // Base rate of new neuron creation (per ms)
#define NEUROGENESIS_ACTIVITY_THRESHOLD 0.8f // Activity threshold for triggering neurogenesis
#define NEUROGENESIS_CAPACITY_FACTOR 1.2f    // Maximum network expansion factor
#define NEUROGENESIS_COOLDOWN_PERIOD 5000.0f // Minimum time between neurogenesis events (ms)

// Neural development stages
#define NEURAL_STAGE_PROGENITOR     0         // Neural progenitor stage
#define NEURAL_STAGE_MIGRATION      1         // Migration stage
#define NEURAL_STAGE_DIFFERENTIATION 2       // Differentiation stage
#define NEURAL_STAGE_INTEGRATION    3         // Integration stage
#define NEURAL_STAGE_MATURE         4         // Mature stage
#define NEURAL_STAGE_SENESCENCE     5         // Senescence stage

// Development time constants (in milliseconds)
#define DEVELOPMENT_MIGRATION_TIME    2000.0f // Time for migration
#define DEVELOPMENT_DIFFERENTIATION_TIME 5000.0f // Time for differentiation
#define DEVELOPMENT_INTEGRATION_TIME  10000.0f // Time for integration
#define DEVELOPMENT_MATURATION_TIME   20000.0f // Time to reach maturity

// Survival and death parameters
#define NEURAL_SURVIVAL_THRESHOLD   0.1f      // Minimum activity for survival
#define NEURAL_DEATH_PROBABILITY    0.00001f  // Base death probability (per ms)
#define NEURAL_COMPETITION_RADIUS   100.0f    // Radius for competitive survival
#define NEURAL_SUPPORT_FACTOR       2.0f      // Activity boost from neighbors

// Specialization parameters
#define SPECIALIZATION_RATE         0.001f    // Rate of functional specialization
#define SPECIALIZATION_THRESHOLD    0.5f      // Threshold for specialization commitment
#define PLASTICITY_CRITICAL_PERIOD  30000.0f  // Critical period for high plasticity (ms)
#define HOMEOSTATIC_TARGET_DENSITY  0.8f      // Target neural density

/**
 * Neural progenitor cell state
 */
struct NeuralProgenitor {
    // Position and movement
    float position_x, position_y, position_z; // 3D position in network space
    float velocity_x, velocity_y, velocity_z; // Migration velocity
    float target_x, target_y, target_z;       // Target destination
    
    // Developmental state
    int development_stage;          // Current developmental stage
    float development_timer;        // Time spent in current stage
    float birth_time;              // Time when progenitor was created
    float maturation_progress;      // Progress towards maturation (0-1)
    
    // Molecular signals
    float growth_factors[8];        // Growth factor concentrations
    float guidance_cues[8];         // Axon guidance cue responses
    float neurotrophins[4];         // Neurotrophin levels (BDNF, NGF, NT3, NT4)
    float morphogens[4];           // Morphogen gradients for patterning
    
    // Cellular properties
    float cell_cycle_phase;        // Current cell cycle phase (0-1)
    float proliferation_potential; // Remaining division potential
    float differentiation_bias;    // Bias towards specific cell types
    float apoptosis_susceptibility; // Susceptibility to programmed cell death
    
    // Target neuron properties
    int target_neuron_type;        // Intended neuron type (exc/inh)
    int target_compartment_count;  // Number of compartments to develop
    float target_excitability;     // Target excitability level
    float target_connectivity;     // Target connectivity degree
    
    // Environmental responsiveness
    float activity_sensor;         // Local activity sensing
    float resource_availability;   // Local resource availability
    float competition_pressure;    // Competition from nearby neurons
    float survival_fitness;        // Current survival fitness
};

/**
 * Developmental trajectory for maturing neurons
 */
struct DevelopmentalTrajectory {
    // Morphological development
    float dendrite_growth_rate[MAX_COMPARTMENTS]; // Growth rate of each dendrite
    float axon_length;                           // Current axon length
    float spine_density[MAX_COMPARTMENTS];       // Spine density on each compartment
    float branch_probability[MAX_COMPARTMENTS];  // Probability of branching
    
    // Electrophysiological maturation
    float membrane_resistance_factor;    // Membrane resistance relative to adult
    float capacitance_factor;           // Membrane capacitance relative to adult
    float excitability_factor;          // Excitability relative to adult
    float plasticity_factor;            // Plasticity level relative to adult
    
    // Ion channel development
    float channel_expression[NUM_RECEPTOR_TYPES]; // Relative ion channel expression
    float channel_maturation[NUM_RECEPTOR_TYPES]; // Ion channel maturation level
    float calcium_dynamics_maturity;              // Calcium system maturity
    
    // Synaptic development
    float synapse_formation_rate;       // Rate of new synapse formation
    float synapse_elimination_rate;     // Rate of synapse elimination
    float synaptic_strength_scaling;    // Scaling factor for synaptic strengths
    float activity_dependent_scaling;   // Activity-dependent strength adjustment
    
    // Functional specialization
    float specialization_commitment[16]; // Commitment to functional roles
    float experience_dependent_tuning;   // Experience-dependent property tuning
    float critical_period_sensitivity;  // Sensitivity during critical periods
    float homeostatic_set_point;       // Target activity level for homeostasis
};

/**
 * Neurogenesis control system
 */
struct NeurogenesisController {
    // Population dynamics
    int current_neuron_count;       // Current number of neurons
    int maximum_neuron_capacity;    // Maximum allowed neurons
    int target_neuron_count;        // Target neuron count based on demand
    float population_growth_rate;   // Current population growth rate
    
    // Activity monitoring
    float network_activity_level;   // Global network activity level
    float local_activity_map[64];   // Local activity levels across regions
    float activity_demand_signal;   // Signal for activity-dependent neurogenesis
    float learning_demand_signal;   // Signal for learning-dependent neurogenesis
    
    // Resource management
    float metabolic_capacity;       // Available metabolic resources
    float space_availability;       // Available physical space
    float competitive_pressure;     // Competition between neurons
    float resource_allocation_efficiency; // Efficiency of resource usage
    
    // Regulation signals
    float neurogenesis_permissive_signal; // Overall permissiveness for neurogenesis
    float apoptosis_regulation_signal;     // Regulation of programmed cell death
    float proliferation_control_signal;   // Control of progenitor proliferation
    float differentiation_timing_signal;  // Control of differentiation timing
    
    // Environmental factors
    float environmental_enrichment;  // Richness of environmental stimulation
    float stress_level;             // Current stress level (inhibits neurogenesis)
    float social_interaction_level; // Social interaction (promotes neurogenesis)
    float novelty_exposure;         // Exposure to novel experiences
    
    // Temporal dynamics
    float circadian_phase;          // Current circadian phase (affects neurogenesis)
    float seasonal_factor;          // Seasonal modulation
    float developmental_clock;      // Overall developmental timing
    float aging_factor;            // Age-related decline in neurogenesis
};

/**
 * CUDA kernel for neurogenesis control and progenitor management
 */
__global__ void neurogenesisControlKernel(
    NeuralProgenitor* progenitors,
    GPUNeuronState* neurons,
    NeurogenesisController* controller,
    DevelopmentalTrajectory* trajectories,
    ValueFunction* value_functions,
    curandState* rng_states,
    float current_time,
    float dt,
    int max_progenitors,
    int current_neuron_count,
    int max_neuron_capacity
) {
    int progenitor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (progenitor_idx >= max_progenitors) return;
    
    NeuralProgenitor& progenitor = progenitors[progenitor_idx];
    curandState local_rng = rng_states[progenitor_idx];
    
    // Skip inactive progenitors
    if (progenitor.development_stage < 0) return;
    
    // ========================================
    // ENVIRONMENTAL SENSING
    // ========================================
    
    // Sense local network activity
    float local_activity = 0.0f;
    float local_neuron_count = 0.0f;
    float local_competition = 0.0f;
    
    // Sample nearby neurons for local environment assessment
    for (int i = 0; i < min(100, current_neuron_count); i++) {
        int neuron_idx = (progenitor_idx * 100 + i) % current_neuron_count;
        GPUNeuronState& neuron = neurons[neuron_idx];
        
        if (neuron.active) {
            // Simple distance calculation (would be better with actual positions)
            float distance = fabsf((float)(neuron_idx - progenitor_idx * 100)) / 100.0f;
            
            if (distance < NEURAL_COMPETITION_RADIUS) {
                local_activity += neuron.activity_level * expf(-distance / 50.0f);
                local_neuron_count += 1.0f;
                local_competition += neuron.activity_level;
            }
        }
    }
    
    progenitor.activity_sensor = local_activity / fmaxf(1.0f, local_neuron_count);
    progenitor.competition_pressure = local_competition / fmaxf(1.0f, local_neuron_count);
    
    // ========================================
    // DEVELOPMENTAL STAGE PROGRESSION
    // ========================================
    
    progenitor.development_timer += dt;
    
    switch (progenitor.development_stage) {
        case NEURAL_STAGE_PROGENITOR:
            // Progenitor stage - decide whether to divide or differentiate
            if (progenitor.development_timer > 1000.0f) { // 1 second minimum
                float differentiation_signal = progenitor.activity_sensor + 
                    controller->differentiation_timing_signal;
                
                if (differentiation_signal > 0.5f || 
                    progenitor.development_timer > 5000.0f) {
                    // Begin migration
                    progenitor.development_stage = NEURAL_STAGE_MIGRATION;
                    progenitor.development_timer = 0.0f;
                    
                    // Set migration target based on activity gradients
                    float target_bias = curand_uniform(&local_rng) * 2.0f - 1.0f;
                    progenitor.target_x = progenitor.position_x + target_bias * 50.0f;
                    progenitor.target_y = progenitor.position_y + target_bias * 50.0f;
                    progenitor.target_z = progenitor.position_z + target_bias * 10.0f;
                }
            }
            break;
            
        case NEURAL_STAGE_MIGRATION:
            // Migration stage - move towards target location
            {
                float dx = progenitor.target_x - progenitor.position_x;
                float dy = progenitor.target_y - progenitor.position_y;
                float dz = progenitor.target_z - progenitor.position_z;
                float distance = sqrtf(dx*dx + dy*dy + dz*dz);
                
                if (distance > 1.0f) {
                    // Continue migration
                    float speed = 0.1f; // Migration speed
                    progenitor.velocity_x = speed * dx / distance;
                    progenitor.velocity_y = speed * dy / distance;
                    progenitor.velocity_z = speed * dz / distance;
                    
                    progenitor.position_x += progenitor.velocity_x * dt;
                    progenitor.position_y += progenitor.velocity_y * dt;
                    progenitor.position_z += progenitor.velocity_z * dt;
                } else {
                    // Reached target - begin differentiation
                    progenitor.development_stage = NEURAL_STAGE_DIFFERENTIATION;
                    progenitor.development_timer = 0.0f;
                }
                
                if (progenitor.development_timer > DEVELOPMENT_MIGRATION_TIME) {
                    // Timeout - force differentiation
                    progenitor.development_stage = NEURAL_STAGE_DIFFERENTIATION;
                    progenitor.development_timer = 0.0f;
                }
            }
            break;
            
        case NEURAL_STAGE_DIFFERENTIATION:
            // Differentiation stage - develop neuron type and initial properties
            {
                float progress = progenitor.development_timer / DEVELOPMENT_DIFFERENTIATION_TIME;
                progenitor.maturation_progress = progress;
                
                // Determine neuron type based on local environment
                if (progenitor.target_neuron_type == 0) { // Not yet determined
                    float inhibitory_bias = progenitor.competition_pressure;
                    if (inhibitory_bias > 0.7f) {
                        progenitor.target_neuron_type = 1; // Inhibitory
                    } else {
                        progenitor.target_neuron_type = 0; // Excitatory
                    }
                }
                
                // Set compartment count based on target function
                if (progenitor.target_compartment_count == 0) {
                    if (progenitor.target_neuron_type == 1) {
                        progenitor.target_compartment_count = 2; // Simple inhibitory
                    } else {
                        progenitor.target_compartment_count = 3 + 
                            (int)(curand_uniform(&local_rng) * 3); // 3-5 compartments
                    }
                }
                
                if (progenitor.development_timer > DEVELOPMENT_DIFFERENTIATION_TIME) {
                    // Begin integration
                    progenitor.development_stage = NEURAL_STAGE_INTEGRATION;
                    progenitor.development_timer = 0.0f;
                }
            }
            break;
            
        case NEURAL_STAGE_INTEGRATION:
            // Integration stage - form initial connections
            {
                float progress = progenitor.development_timer / DEVELOPMENT_INTEGRATION_TIME;
                progenitor.maturation_progress = 0.2f + 0.6f * progress;
                
                // This stage would involve creating synaptic connections
                // Implementation would require coordination with synaptogenesis system
                
                if (progenitor.development_timer > DEVELOPMENT_INTEGRATION_TIME) {
                    // Ready for maturation - create actual neuron
                    progenitor.development_stage = NEURAL_STAGE_MATURE;
                    progenitor.development_timer = 0.0f;
                }
            }
            break;
            
        case NEURAL_STAGE_MATURE:
            // Mature stage - neuron is fully functional
            progenitor.maturation_progress = 1.0f;
            
            // Check for senescence
            float age = current_time - progenitor.birth_time;
            if (age > 3600000.0f) { // 1 hour = very old for simulation
                float senescence_probability = (age - 3600000.0f) / 3600000.0f;
                if (curand_uniform(&local_rng) < senescence_probability * dt * 0.00001f) {
                    progenitor.development_stage = NEURAL_STAGE_SENESCENCE;
                }
            }
            break;
            
        case NEURAL_STAGE_SENESCENCE:
            // Senescence stage - prepare for death
            progenitor.maturation_progress = fmaxf(0.0f, progenitor.maturation_progress - 0.001f * dt);
            
            if (progenitor.maturation_progress < 0.1f || 
                curand_uniform(&local_rng) < NEURAL_DEATH_PROBABILITY * dt) {
                // Mark for death
                progenitor.development_stage = -1; // Inactive marker
            }
            break;
    }
    
    // ========================================
    // SURVIVAL FITNESS COMPUTATION
    // ========================================
    
    // Compute survival fitness based on multiple factors
    float activity_fitness = fminf(1.0f, progenitor.activity_sensor / NEURAL_SURVIVAL_THRESHOLD);
    float competition_fitness = 1.0f / (1.0f + progenitor.competition_pressure);
    float resource_fitness = progenitor.resource_availability;
    float developmental_fitness = (progenitor.development_stage >= NEURAL_STAGE_INTEGRATION) ? 1.0f : 0.5f;
    
    progenitor.survival_fitness = (activity_fitness + competition_fitness + 
                                  resource_fitness + developmental_fitness) / 4.0f;
    
    // ========================================
    // GROWTH FACTOR DYNAMICS
    // ========================================
    
    // Update growth factor concentrations based on local environment
    for (int i = 0; i < 8; i++) {
        float target_concentration = 0.5f + 0.3f * progenitor.activity_sensor;
        if (i < 4) { // First 4 are activity-dependent
            target_concentration += 0.2f * progenitor.activity_sensor;
        } else { // Last 4 are competition-dependent
            target_concentration -= 0.2f * progenitor.competition_pressure;
        }
        
        progenitor.growth_factors[i] += (target_concentration - progenitor.growth_factors[i]) * 
                                       0.001f * dt;
        progenitor.growth_factors[i] = fmaxf(0.0f, fminf(1.0f, progenitor.growth_factors[i]));
    }
    
    // Update neurotrophin levels
    for (int i = 0; i < 4; i++) {
        float activity_dependent_neurotrophin = 0.3f + 0.4f * progenitor.activity_sensor;
        progenitor.neurotrophins[i] += (activity_dependent_neurotrophin - progenitor.neurotrophins[i]) * 
                                      0.01f * dt;
        progenitor.neurotrophins[i] = fmaxf(0.0f, fminf(1.0f, progenitor.neurotrophins[i]));
    }
    
    // Update RNG state
    rng_states[progenitor_idx] = local_rng;
}

/**
 * CUDA kernel for creating new neural progenitors based on demand
 */
__global__ void neurogenesisSpawningKernel(
    NeuralProgenitor* progenitors,
    GPUNeuronState* neurons,
    NeurogenesisController* controller,
    ValueFunction* value_functions,
    curandState* rng_states,
    float current_time,
    float dt,
    int max_progenitors,
    int current_neuron_count,
    int* new_progenitor_count
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx != 0) return; // Only one thread handles spawning decisions
    
    // ========================================
    // ASSESS NEUROGENESIS DEMAND
    // ========================================
    
    // Calculate network activity level
    float total_activity = 0.0f;
    int active_neurons = 0;
    
    for (int i = 0; i < current_neuron_count; i++) {
        if (neurons[i].active) {
            total_activity += neurons[i].activity_level;
            active_neurons++;
        }
    }
    
    float average_activity = (active_neurons > 0) ? total_activity / active_neurons : 0.0f;
    controller->network_activity_level = average_activity;
    
    // ========================================
    // DETERMINE NEUROGENESIS NEED
    // ========================================
    
    bool needs_neurogenesis = false;
    float neurogenesis_urgency = 0.0f;
    
    // Activity-based neurogenesis trigger
    if (average_activity > NEUROGENESIS_ACTIVITY_THRESHOLD) {
        needs_neurogenesis = true;
        neurogenesis_urgency += (average_activity - NEUROGENESIS_ACTIVITY_THRESHOLD) / 
                               (1.0f - NEUROGENESIS_ACTIVITY_THRESHOLD);
    }
    
    // Learning-based neurogenesis trigger
    float learning_pressure = 0.0f;
    for (int i = 0; i < min(100, current_neuron_count / 100); i++) {
        if (value_functions[i].prediction_uncertainty > 0.5f) {
            learning_pressure += value_functions[i].prediction_uncertainty;
        }
    }
    learning_pressure /= 100.0f;
    
    if (learning_pressure > 0.3f) {
        needs_neurogenesis = true;
        neurogenesis_urgency += learning_pressure;
    }
    
    // Capacity check
    if (current_neuron_count >= controller->maximum_neuron_capacity) {
        needs_neurogenesis = false;
    }
    
    // Cooldown check
    static float last_neurogenesis_time = 0.0f;
    if (current_time - last_neurogenesis_time < NEUROGENESIS_COOLDOWN_PERIOD) {
        needs_neurogenesis = false;
    }
    
    // ========================================
    // CREATE NEW PROGENITORS
    // ========================================
    
    if (needs_neurogenesis) {
        int progenitors_to_create = (int)(neurogenesis_urgency * 5.0f); // Up to 5 new progenitors
        progenitors_to_create = min(progenitors_to_create, 10); // Hard limit
        
        curandState local_rng = rng_states[0];
        int created_count = 0;
        
        for (int p = 0; p < max_progenitors && created_count < progenitors_to_create; p++) {
            if (progenitors[p].development_stage < 0) { // Found inactive slot
                // Initialize new progenitor
                NeuralProgenitor& new_progenitor = progenitors[p];
                
                new_progenitor.development_stage = NEURAL_STAGE_PROGENITOR;
                new_progenitor.development_timer = 0.0f;
                new_progenitor.birth_time = current_time;
                new_progenitor.maturation_progress = 0.0f;
                
                // Random initial position
                new_progenitor.position_x = curand_uniform(&local_rng) * 1000.0f;
                new_progenitor.position_y = curand_uniform(&local_rng) * 1000.0f;
                new_progenitor.position_z = curand_uniform(&local_rng) * 100.0f;
                
                // Initialize velocities
                new_progenitor.velocity_x = 0.0f;
                new_progenitor.velocity_y = 0.0f;
                new_progenitor.velocity_z = 0.0f;
                
                // Initialize cellular properties
                new_progenitor.cell_cycle_phase = curand_uniform(&local_rng);
                new_progenitor.proliferation_potential = 3.0f + curand_uniform(&local_rng) * 2.0f;
                new_progenitor.differentiation_bias = curand_uniform(&local_rng);
                new_progenitor.apoptosis_susceptibility = 0.1f + curand_uniform(&local_rng) * 0.1f;
                
                // Target properties (will be determined during development)
                new_progenitor.target_neuron_type = 0; // TBD
                new_progenitor.target_compartment_count = 0; // TBD
                new_progenitor.target_excitability = 0.5f + curand_uniform(&local_rng) * 0.5f;
                new_progenitor.target_connectivity = 10.0f + curand_uniform(&local_rng) * 40.0f;
                
                // Initialize growth factors and signals
                for (int i = 0; i < 8; i++) {
                    new_progenitor.growth_factors[i] = 0.3f + curand_uniform(&local_rng) * 0.4f;
                    if (i < 8) new_progenitor.guidance_cues[i] = curand_uniform(&local_rng);
                }
                
                for (int i = 0; i < 4; i++) {
                    new_progenitor.neurotrophins[i] = 0.4f + curand_uniform(&local_rng) * 0.2f;
                    new_progenitor.morphogens[i] = curand_uniform(&local_rng);
                }
                
                // Environmental sensing
                new_progenitor.activity_sensor = 0.0f;
                new_progenitor.resource_availability = 0.8f;
                new_progenitor.competition_pressure = 0.0f;
                new_progenitor.survival_fitness = 1.0f;
                
                created_count++;
            }
        }
        
        *new_progenitor_count = created_count;
        last_neurogenesis_time = current_time;
        rng_states[0] = local_rng;
        
        // Update controller state
        controller->neurogenesis_permissive_signal = neurogenesis_urgency;
        controller->target_neuron_count = current_neuron_count + created_count;
    } else {
        *new_progenitor_count = 0;
        controller->neurogenesis_permissive_signal *= 0.99f; // Decay signal
    }
}

/**
 * Host function to launch neurogenesis system
 */
void launchNeurogenesisSystem(
    NeuralProgenitor* d_progenitors,
    GPUNeuronState* d_neurons,
    NeurogenesisController* d_controller,
    DevelopmentalTrajectory* d_trajectories,
    ValueFunction* d_value_functions,
    curandState* d_rng_states,
    int* d_new_progenitor_count,
    float current_time,
    float dt,
    int max_progenitors,
    int current_neuron_count,
    int max_neuron_capacity
) {
    // Launch progenitor development kernel
    {
        dim3 block(256);
        dim3 grid((max_progenitors + block.x - 1) / block.x);
        
        neurogenesisControlKernel<<<grid, block>>>(
            d_progenitors, d_neurons, d_controller, d_trajectories,
            d_value_functions, d_rng_states, current_time, dt,
            max_progenitors, current_neuron_count, max_neuron_capacity
        );
    }
    
    // Launch neurogenesis spawning kernel
    {
        dim3 block(1);
        dim3 grid(1);
        
        neurogenesisSpawningKernel<<<grid, block>>>(
            d_progenitors, d_neurons, d_controller, d_value_functions,
            d_rng_states, current_time, dt, max_progenitors, 
            current_neuron_count, d_new_progenitor_count
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in neurogenesis system: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

#endif // DYNAMIC_NEUROGENESIS_FRAMEWORK_H