// Phase3IntegrationFramework.h
#ifndef PHASE3_INTEGRATION_FRAMEWORK_H
#define PHASE3_INTEGRATION_FRAMEWORK_H

#include "EnhancedSTDPFramework.h"
#include "AdvancedReinforcementLearning.h"
#include "DynamicNeurogenesisFramework.h"
#include "DynamicSynaptogenesisFramework.h"
#include "NeuralPruningFramework.h"
#include "HomeostaticRegulationSystem.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <vector>
#include <memory>

/**
 * Complete Phase 3 integration framework implementing:
 * - Unified dynamic neural network with all Phase 3 features
 * - Coordinated plasticity, neurogenesis, and pruning
 * - Adaptive learning with homeostatic regulation
 * - Real-time performance monitoring and optimization
 * - Scalable architecture for large networks
 */

// ========================================
// INTEGRATION CONSTANTS
// ========================================

// System coordination parameters
#define INTEGRATION_TIMESTEP_MS         0.1f      // Base integration timestep
#define ADAPTIVE_TIMESTEP_ENABLED       true      // Enable adaptive timestep
#define COORDINATION_UPDATE_INTERVAL    10        // Steps between coordination updates
#define PERFORMANCE_MONITORING_INTERVAL 1000     // Steps between performance checks

// Resource management
#define MEMORY_POOL_SIZE_MB            1024      // Memory pool size in MB
#define COMPUTATION_BUDGET_MS          10.0f     // Maximum computation time per step
#define DYNAMIC_LOAD_BALANCING         true      // Enable dynamic load balancing
#define MEMORY_COMPACTION_THRESHOLD    0.8f      // Trigger compaction at 80% fragmentation

// Learning coordination
#define MULTI_TIMESCALE_COORDINATION   true      // Coordinate multiple timescales
#define LEARNING_RATE_ADAPTATION       true      // Adaptive learning rates
#define CURIOSITY_DRIVEN_EXPLORATION   true      // Enable curiosity-driven learning
#define EXPERIENCE_REPLAY_ENABLED      true      // Enable experience replay

/**
 * System state tracking for coordination
 */
struct SystemState {
    // Network topology metrics
    int current_neuron_count;           // Current number of active neurons
    int current_synapse_count;          // Current number of active synapses
    float network_density;              // Connection density
    float small_world_coefficient;      // Small-world network coefficient
    
    // Activity and performance metrics
    float global_activity_level;        // Overall network activity
    float learning_progress_rate;       // Rate of learning progress
    float exploration_exploitation_ratio; // Exploration vs exploitation balance
    float network_efficiency;           // Computational efficiency
    
    // Resource utilization
    float memory_utilization;           // Memory usage percentage
    float computation_utilization;      // Computation usage percentage
    float energy_efficiency;            // Energy efficiency metric
    float resource_fragmentation;       // Resource fragmentation level
    
    // Adaptation metrics
    float plasticity_rate;              // Rate of synaptic changes
    float structural_change_rate;       // Rate of structural modifications
    float homeostatic_error;            // Deviation from homeostatic targets
    float stability_index;              // Network stability measure
    
    // Learning and performance
    float prediction_accuracy;          // Prediction accuracy
    float reward_prediction_error;      // RPE magnitude
    float novelty_detection_rate;       // Rate of novelty detection
    float competence_progression;       // Rate of skill acquisition
};

/**
 * Coordination controller managing system-wide dynamics
 */
struct CoordinationController {
    // Temporal coordination
    float current_time;                 // Current simulation time
    float adaptive_timestep;            // Current adaptive timestep
    int coordination_cycle;             // Current coordination cycle
    float temporal_scale_factor;        // Temporal scaling factor
    
    // Process scheduling
    bool plasticity_update_due;         // Plasticity update scheduled
    bool neurogenesis_update_due;       // Neurogenesis update scheduled
    bool pruning_update_due;            // Pruning update scheduled
    bool homeostasis_update_due;        // Homeostasis update scheduled
    
    // Load balancing
    float plasticity_load;              // Computational load of plasticity
    float neurogenesis_load;            // Computational load of neurogenesis
    float pruning_load;                 // Computational load of pruning
    float homeostasis_load;             // Computational load of homeostasis
    
    // Adaptive scheduling
    int plasticity_update_interval;     // Interval for plasticity updates
    int neurogenesis_update_interval;   // Interval for neurogenesis updates
    int pruning_update_interval;        // Interval for pruning updates
    int homeostasis_update_interval;    // Interval for homeostasis updates
    
    // Resource allocation
    float memory_allocation[4];         // Memory allocation per subsystem
    float computation_allocation[4];    // Computation allocation per subsystem
    float priority_weights[4];          // Priority weights for subsystems
    
    // Emergency controls
    bool emergency_pruning_active;      // Emergency pruning activated
    bool learning_rate_reduction;       // Learning rate reduction active
    bool structural_modifications_paused; // Structural modifications paused
    bool stability_protection_mode;     // Stability protection activated
};

/**
 * Complete dynamic neural network class integrating all Phase 3 features
 */
class DynamicNeuralNetwork {
public:
    // ========================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================
    
    DynamicNeuralNetwork(int initial_neurons, int initial_synapses, int max_capacity_neurons, int max_capacity_synapses);
    ~DynamicNeuralNetwork();
    
    bool initialize();
    bool initializePhase3Systems();
    bool validateInitialization();
    
    // ========================================
    // MAIN SIMULATION INTERFACE
    // ========================================
    
    void simulationStep(float dt, float current_time);
    void adaptiveSimulationStep(float target_dt, float current_time);
    void multiStepSimulation(int num_steps, float dt, float start_time);
    
    // ========================================
    // LEARNING AND ADAPTATION
    // ========================================
    
    void setRewardSignal(float reward);
    void setEnvironmentalFeatures(const std::vector<float>& features);
    void enableCuriosityDrivenLearning(bool enable);
    void setLearningRateAdaptation(bool enable);
    
    // ========================================
    // DYNAMIC STRUCTURE CONTROL
    // ========================================
    
    void enableNeurogenesis(bool enable);
    void enableSynaptogenesis(bool enable);
    void enablePruning(bool enable);
    void setStructuralPlasticityRate(float rate);
    
    // ========================================
    // HOMEOSTATIC CONTROL
    // ========================================
    
    void setHomeostaticTargets(float activity_target, float connectivity_target);
    void enableHomeostaticRegulation(bool enable);
    void setStabilityProtection(bool enable);
    
    // ========================================
    // MONITORING AND ANALYSIS
    // ========================================
    
    SystemState getSystemState() const;
    void getDetailedStatistics(std::vector<float>& stats) const;
    void exportNetworkStructure(const std::string& filename) const;
    void generatePerformanceReport(const std::string& filename) const;
    
    // ========================================
    // PERFORMANCE OPTIMIZATION
    // ========================================
    
    void enableAdaptiveTimestep(bool enable);
    void setComputationBudget(float budget_ms);
    void enableMemoryOptimization(bool enable);
    void optimizeForLatency();
    void optimizeForThroughput();
    
    // ========================================
    // DEVICE MEMORY ACCESS
    // ========================================
    
    GPUNeuronState* getDeviceNeurons() const { return d_neurons_; }
    GPUSynapse* getDeviceSynapses() const { return d_synapses_; }
    
private:
    // ========================================
    // DEVICE MEMORY MANAGEMENT
    // ========================================
    
    // Core neural structures
    GPUNeuronState* d_neurons_;
    GPUSynapse* d_synapses_;
    curandState* d_rng_states_;
    
    // Phase 3 component states
    PlasticityState* d_plasticity_states_;
    DopamineNeuron* d_dopamine_neurons_;
    ValueFunction* d_value_functions_;
    ActorCriticState* d_actor_critic_states_;
    CuriositySystem* d_curiosity_systems_;
    
    // Dynamic structure components
    NeuralProgenitor* d_neural_progenitors_;
    DevelopmentalTrajectory* d_developmental_trajectories_;
    SynapticProgenitor* d_synaptic_progenitors_;
    SynapticCompetition* d_synaptic_competition_;
    
    // Pruning and regulation
    PruningAssessment* d_neuron_assessments_;
    PruningAssessment* d_synapse_assessments_;
    CompetitiveElimination* d_neuron_competition_;
    CompetitiveElimination* d_synapse_competition_;
    
    // Homeostatic regulation
    NeuralHomeostasis* d_neural_homeostasis_;
    SynapticHomeostasis* d_synaptic_homeostasis_;
    NetworkHomeostasis* d_network_homeostasis_;
    
    // Control and coordination
    STDPRuleConfig* d_stdp_config_;
    NeurogenesisController* d_neurogenesis_controller_;
    SynaptogenesisController* d_synaptogenesis_controller_;
    PruningController* d_pruning_controller_;
    CoordinationController* d_coordination_controller_;
    
    // Environmental interface
    float* d_global_reward_signal_;
    float* d_environmental_features_;
    float* d_global_neuromodulators_;
    
    // Performance monitoring
    float* d_performance_metrics_;
    int* d_active_element_counts_;
    
    // ========================================
    // SYSTEM CONFIGURATION
    // ========================================
    
    int max_neurons_;
    int max_synapses_;
    int max_progenitors_;
    int current_neurons_;
    int current_synapses_;
    
    float current_time_;
    float adaptive_timestep_;
    int simulation_step_count_;
    
    bool phase3_initialized_;
    bool homeostatic_regulation_enabled_;
    bool structural_plasticity_enabled_;
    bool curiosity_learning_enabled_;
    
    // ========================================
    // PERFORMANCE MONITORING
    // ========================================
    
    std::chrono::high_resolution_clock::time_point last_performance_check_;
    std::vector<float> performance_history_;
    float average_step_time_ms_;
    float peak_memory_usage_mb_;
    
    // ========================================
    // PRIVATE METHODS
    // ========================================
    
    // Initialization helpers
    bool allocateDeviceMemory();
    void deallocateDeviceMemory();
    bool initializeRandomStates();
    bool initializeControllers();
    
    // Simulation coordination
    void coordinateSubsystems(float dt);
    void updateCoordinationController(float dt);
    void adaptiveScheduling();
    void resourceLoadBalancing();
    
    // Performance optimization
    void performanceMonitoring();
    void memoryCompaction();
    void adaptiveTimestepControl(float dt);
    void emergencyStabilization();
    
    // Subsystem updates
    void updatePlasticity(float dt);
    void updateNeurogenesis(float dt);
    void updateSynaptogenesis(float dt);
    void updatePruning(float dt);
    void updateHomeostasis(float dt);
    void updateReinforcementLearning(float dt);
    
    // Utility functions
    void updateSystemState();
    void validateSystemIntegrity();
    bool checkResourceLimits();
    void handleSystemErrors();
};

// ========================================
// IMPLEMENTATION
// ========================================

DynamicNeuralNetwork::DynamicNeuralNetwork(
    int initial_neurons, 
    int initial_synapses, 
    int max_capacity_neurons, 
    int max_capacity_synapses
) : max_neurons_(max_capacity_neurons)
  , max_synapses_(max_capacity_synapses)
  , max_progenitors_(max_capacity_neurons / 10)  // 10% progenitor capacity
  , current_neurons_(initial_neurons)
  , current_synapses_(initial_synapses)
  , current_time_(0.0f)
  , adaptive_timestep_(INTEGRATION_TIMESTEP_MS)
  , simulation_step_count_(0)
  , phase3_initialized_(false)
  , homeostatic_regulation_enabled_(true)
  , structural_plasticity_enabled_(true)
  , curiosity_learning_enabled_(true)
  , average_step_time_ms_(0.0f)
  , peak_memory_usage_mb_(0.0f)
{
    // Initialize all device pointers to nullptr
    d_neurons_ = nullptr;
    d_synapses_ = nullptr;
    d_rng_states_ = nullptr;
    d_plasticity_states_ = nullptr;
    d_dopamine_neurons_ = nullptr;
    d_value_functions_ = nullptr;
    d_actor_critic_states_ = nullptr;
    d_curiosity_systems_ = nullptr;
    d_neural_progenitors_ = nullptr;
    d_developmental_trajectories_ = nullptr;
    d_synaptic_progenitors_ = nullptr;
    d_synaptic_competition_ = nullptr;
    d_neuron_assessments_ = nullptr;
    d_synapse_assessments_ = nullptr;
    d_neuron_competition_ = nullptr;
    d_synapse_competition_ = nullptr;
    d_neural_homeostasis_ = nullptr;
    d_synaptic_homeostasis_ = nullptr;
    d_network_homeostasis_ = nullptr;
    d_stdp_config_ = nullptr;
    d_neurogenesis_controller_ = nullptr;
    d_synaptogenesis_controller_ = nullptr;
    d_pruning_controller_ = nullptr;
    d_coordination_controller_ = nullptr;
    d_global_reward_signal_ = nullptr;
    d_environmental_features_ = nullptr;
    d_global_neuromodulators_ = nullptr;
    d_performance_metrics_ = nullptr;
    d_active_element_counts_ = nullptr;
}

DynamicNeuralNetwork::~DynamicNeuralNetwork() {
    deallocateDeviceMemory();
}

bool DynamicNeuralNetwork::initialize() {
    printf("Initializing Dynamic Neural Network (Phase 3)...\n");
    printf("Target capacity: %d neurons, %d synapses\n", max_neurons_, max_synapses_);
    
    // Allocate device memory
    if (!allocateDeviceMemory()) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return false;
    }
    
    // Initialize random number generators
    if (!initializeRandomStates()) {
        fprintf(stderr, "Failed to initialize random states\n");
        return false;
    }
    
    // Initialize control systems
    if (!initializeControllers()) {
        fprintf(stderr, "Failed to initialize controllers\n");
        return false;
    }
    
    // Initialize Phase 3 systems
    if (!initializePhase3Systems()) {
        fprintf(stderr, "Failed to initialize Phase 3 systems\n");
        return false;
    }
    
    // Validate initialization
    if (!validateInitialization()) {
        fprintf(stderr, "Initialization validation failed\n");
        return false;
    }
    
    phase3_initialized_ = true;
    last_performance_check_ = std::chrono::high_resolution_clock::now();
    
    printf("Dynamic Neural Network initialization complete\n");
    return true;
}

void DynamicNeuralNetwork::simulationStep(float dt, float current_time) {
    auto step_start = std::chrono::high_resolution_clock::now();
    
    current_time_ = current_time;
    simulation_step_count_++;
    
    // ========================================
    // COORDINATION AND SCHEDULING
    // ========================================
    
    if (simulation_step_count_ % COORDINATION_UPDATE_INTERVAL == 0) {
        coordinateSubsystems(dt);
    }
    
    // ========================================
    // CORE NEURAL DYNAMICS (ALWAYS ACTIVE)
    // ========================================
    
    // Update ion channel dynamics (from Phase 2)
    launchEnhancedSynapticInput(
        d_synapses_, d_neurons_, nullptr, // No spike events for now
        nullptr, nullptr, // No current arrays for now
        nullptr, nullptr, // No synapse mapping for now
        current_synapses_, current_neurons_, 0,
        current_time, dt
    );
    
    // Update calcium dynamics
    launchCalciumDynamics(d_neurons_, dt, current_neurons_);
    
    // Update neuron voltages with enhanced RK4
    dim3 block(256);
    dim3 grid((current_neurons_ + block.x - 1) / block.x);
    enhancedRK4NeuronUpdateKernel<<<grid, block>>>(
        d_neurons_, dt, current_time, current_neurons_
    );
    
    // ========================================
    // PHASE 3 SUBSYSTEM UPDATES
    // ========================================
    
    CoordinationController coord_state;
    cudaMemcpy(&coord_state, d_coordination_controller_, sizeof(CoordinationController), 
               cudaMemcpyDeviceToHost);
    
    // Update plasticity (high frequency)
    if (coord_state.plasticity_update_due) {
        updatePlasticity(dt);
    }
    
    // Update reinforcement learning (high frequency)
    updateReinforcementLearning(dt);
    
    // Update homeostasis (medium frequency)
    if (coord_state.homeostasis_update_due) {
        updateHomeostasis(dt);
    }
    
    // Update neurogenesis (low frequency)
    if (coord_state.neurogenesis_update_due) {
        updateNeurogenesis(dt);
    }
    
    // Update synaptogenesis (low frequency)
    if (coord_state.neurogenesis_update_due) { // Same schedule as neurogenesis
        updateSynaptogenesis(dt);
    }
    
    // Update pruning (low frequency)
    if (coord_state.pruning_update_due) {
        updatePruning(dt);
    }
    
    // ========================================
    // SYSTEM MONITORING AND OPTIMIZATION
    // ========================================
    
    if (simulation_step_count_ % PERFORMANCE_MONITORING_INTERVAL == 0) {
        performanceMonitoring();
        updateSystemState();
        
        // Check for emergency interventions
        if (checkResourceLimits()) {
            emergencyStabilization();
        }
    }
    
    // ========================================
    // ADAPTIVE TIMESTEP CONTROL
    // ========================================
    
    if (ADAPTIVE_TIMESTEP_ENABLED) {
        adaptiveTimestepControl(dt);
    }
    
    // ========================================
    // PERFORMANCE TRACKING
    // ========================================
    
    auto step_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
    float step_time_ms = duration.count() / 1000.0f;
    
    // Update running average
    average_step_time_ms_ = average_step_time_ms_ * 0.99f + step_time_ms * 0.01f;
    
    // Check computation budget
    if (step_time_ms > COMPUTATION_BUDGET_MS) {
        // Exceeded budget - trigger adaptive scheduling
        adaptiveScheduling();
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in simulation step: %s\n", cudaGetErrorString(err));
        handleSystemErrors();
    }
    
    cudaDeviceSynchronize();
}

void DynamicNeuralNetwork::coordinateSubsystems(float dt) {
    // Update coordination controller
    updateCoordinationController(dt);
    
    // Determine which subsystems need updates
    CoordinationController coord_state;
    cudaMemcpy(&coord_state, d_coordination_controller_, sizeof(CoordinationController), 
               cudaMemcpyDeviceToHost);
    
    // Schedule updates based on intervals and load
    coord_state.plasticity_update_due = 
        (simulation_step_count_ % coord_state.plasticity_update_interval == 0);
    coord_state.homeostasis_update_due = 
        (simulation_step_count_ % coord_state.homeostasis_update_interval == 0);
    coord_state.neurogenesis_update_due = 
        (simulation_step_count_ % coord_state.neurogenesis_update_interval == 0);
    coord_state.pruning_update_due = 
        (simulation_step_count_ % coord_state.pruning_update_interval == 0);
    
    // Copy updated scheduling back
    cudaMemcpy(d_coordination_controller_, &coord_state, sizeof(CoordinationController), 
               cudaMemcpyHostToDevice);
}

void DynamicNeuralNetwork::updatePlasticity(float dt) {
    // Enhanced STDP with multi-factor plasticity
    launchEnhancedSTDP(
        d_synapses_, d_neurons_, d_plasticity_states_, d_stdp_config_,
        d_global_neuromodulators_, current_time_, dt, current_synapses_
    );
}

void DynamicNeuralNetwork::updateReinforcementLearning(float dt) {
    // Advanced reinforcement learning with RPE and curiosity
    launchAdvancedReinforcementLearning(
        d_dopamine_neurons_, d_value_functions_, d_actor_critic_states_,
        d_curiosity_systems_, d_neurons_, d_global_reward_signal_,
        d_environmental_features_, current_time_, dt,
        current_neurons_ / 100, current_neurons_  // Sample of dopamine neurons
    );
}

void DynamicNeuralNetwork::updateNeurogenesis(float dt) {
    // Dynamic neurogenesis system
    launchNeurogenesisSystem(
        d_neural_progenitors_, d_neurons_, d_neurogenesis_controller_,
        d_developmental_trajectories_, d_value_functions_, d_rng_states_,
        d_active_element_counts_, current_time_, dt, max_progenitors_,
        current_neurons_, max_neurons_
    );
    
    // Check for new neurons created
    int new_neuron_count;
    cudaMemcpy(&new_neuron_count, d_active_element_counts_, sizeof(int), 
               cudaMemcpyDeviceToHost);
    
    if (new_neuron_count > 0) {
        current_neurons_ = min(max_neurons_, current_neurons_ + new_neuron_count);
        printf("Neurogenesis: %d new neurons created (total: %d)\n", 
               new_neuron_count, current_neurons_);
    }
}

void DynamicNeuralNetwork::updateSynaptogenesis(float dt) {
    // Dynamic synaptogenesis system
    launchSynaptogenesisSystem(
        d_synaptic_progenitors_, d_neurons_, d_synapses_, d_synaptic_competition_,
        d_synaptogenesis_controller_, d_rng_states_, current_time_, dt,
        max_progenitors_, current_neurons_, current_synapses_
    );
}

void DynamicNeuralNetwork::updatePruning(float dt) {
    // Neural pruning and elimination
    launchNeuralPruningSystem(
        d_neurons_, d_synapses_, d_neuron_assessments_, d_synapse_assessments_,
        d_neuron_competition_, d_synapse_competition_, d_pruning_controller_,
        d_value_functions_, d_rng_states_, current_time_, dt,
        current_neurons_, current_synapses_
    );
}

void DynamicNeuralNetwork::updateHomeostasis(float dt) {
    // Homeostatic regulation system
    launchHomeostaticRegulationSystem(
        d_neurons_, d_synapses_, d_neural_homeostasis_, d_synaptic_homeostasis_,
        d_network_homeostasis_, d_plasticity_states_, d_value_functions_,
        current_time_, dt, current_neurons_, current_synapses_
    );
}

SystemState DynamicNeuralNetwork::getSystemState() const {
    SystemState state;
    
    // Copy network state from device
    NetworkHomeostasis network_state;
    cudaMemcpy(&network_state, d_network_homeostasis_, sizeof(NetworkHomeostasis), 
               cudaMemcpyDeviceToHost);
    
    // Fill system state structure
    state.current_neuron_count = current_neurons_;
    state.current_synapse_count = current_synapses_;
    state.network_density = (float)current_synapses_ / (current_neurons_ * current_neurons_);
    state.global_activity_level = network_state.global_activity_level;
    state.network_efficiency = network_state.allocation_efficiency;
    state.stability_index = network_state.stability_index;
    state.homeostatic_error = network_state.activity_regulation_error;
    
    // Performance metrics
    state.memory_utilization = peak_memory_usage_mb_ / MEMORY_POOL_SIZE_MB;
    state.computation_utilization = average_step_time_ms_ / COMPUTATION_BUDGET_MS;
    
    return state;
}

bool DynamicNeuralNetwork::allocateDeviceMemory() {
    cudaError_t err;
    size_t total_memory = 0;
    
    printf("Allocating device memory for dynamic neural network...\n");
    
    // Core neural structures
    err = cudaMalloc(&d_neurons_, max_neurons_ * sizeof(GPUNeuronState));
    if (err != cudaSuccess) return false;
    total_memory += max_neurons_ * sizeof(GPUNeuronState);
    
    err = cudaMalloc(&d_synapses_, max_synapses_ * sizeof(GPUSynapse));
    if (err != cudaSuccess) return false;
    total_memory += max_synapses_ * sizeof(GPUSynapse);
    
    err = cudaMalloc(&d_rng_states_, max_neurons_ * sizeof(curandState));
    if (err != cudaSuccess) return false;
    total_memory += max_neurons_ * sizeof(curandState);
    
    // Phase 3 plasticity components
    err = cudaMalloc(&d_plasticity_states_, max_synapses_ * sizeof(PlasticityState));
    if (err != cudaSuccess) return false;
    total_memory += max_synapses_ * sizeof(PlasticityState);
    
    // Reinforcement learning components
    int num_dopamine_neurons = max_neurons_ / 100;  // 1% dopamine neurons
    err = cudaMalloc(&d_dopamine_neurons_, num_dopamine_neurons * sizeof(DopamineNeuron));
    if (err != cudaSuccess) return false;
    total_memory += num_dopamine_neurons * sizeof(DopamineNeuron);
    
    err = cudaMalloc(&d_value_functions_, num_dopamine_neurons * sizeof(ValueFunction));
    if (err != cudaSuccess) return false;
    total_memory += num_dopamine_neurons * sizeof(ValueFunction);
    
    err = cudaMalloc(&d_actor_critic_states_, num_dopamine_neurons * sizeof(ActorCriticState));
    if (err != cudaSuccess) return false;
    total_memory += num_dopamine_neurons * sizeof(ActorCriticState);
    
    err = cudaMalloc(&d_curiosity_systems_, num_dopamine_neurons * sizeof(CuriositySystem));
    if (err != cudaSuccess) return false;
    total_memory += num_dopamine_neurons * sizeof(CuriositySystem);
    
    // Dynamic structure components
    err = cudaMalloc(&d_neural_progenitors_, max_progenitors_ * sizeof(NeuralProgenitor));
    if (err != cudaSuccess) return false;
    total_memory += max_progenitors_ * sizeof(NeuralProgenitor);
    
    err = cudaMalloc(&d_developmental_trajectories_, max_progenitors_ * sizeof(DevelopmentalTrajectory));
    if (err != cudaSuccess) return false;
    total_memory += max_progenitors_ * sizeof(DevelopmentalTrajectory);
    
    err = cudaMalloc(&d_synaptic_progenitors_, max_progenitors_ * sizeof(SynapticProgenitor));
    if (err != cudaSuccess) return false;
    total_memory += max_progenitors_ * sizeof(SynapticProgenitor);
    
    err = cudaMalloc(&d_synaptic_competition_, max_progenitors_ * sizeof(SynapticCompetition));
    if (err != cudaSuccess) return false;
    total_memory += max_progenitors_ * sizeof(SynapticCompetition);
    
    // Pruning components
    err = cudaMalloc(&d_neuron_assessments_, max_neurons_ * sizeof(PruningAssessment));
    if (err != cudaSuccess) return false;
    total_memory += max_neurons_ * sizeof(PruningAssessment);
    
    err = cudaMalloc(&d_synapse_assessments_, max_synapses_ * sizeof(PruningAssessment));
    if (err != cudaSuccess) return false;
    total_memory += max_synapses_ * sizeof(PruningAssessment);
    
    err = cudaMalloc(&d_neuron_competition_, max_neurons_ * sizeof(CompetitiveElimination));
    if (err != cudaSuccess) return false;
    total_memory += max_neurons_ * sizeof(CompetitiveElimination);
    
    err = cudaMalloc(&d_synapse_competition_, max_synapses_ * sizeof(CompetitiveElimination));
    if (err != cudaSuccess) return false;
    total_memory += max_synapses_ * sizeof(CompetitiveElimination);
    
    // Homeostatic regulation components
    err = cudaMalloc(&d_neural_homeostasis_, max_neurons_ * sizeof(NeuralHomeostasis));
    if (err != cudaSuccess) return false;
    total_memory += max_neurons_ * sizeof(NeuralHomeostasis);
    
    err = cudaMalloc(&d_synaptic_homeostasis_, max_synapses_ * sizeof(SynapticHomeostasis));
    if (err != cudaSuccess) return false;
    total_memory += max_synapses_ * sizeof(SynapticHomeostasis);
    
    err = cudaMalloc(&d_network_homeostasis_, sizeof(NetworkHomeostasis));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(NetworkHomeostasis);
    
    // Control structures
    err = cudaMalloc(&d_stdp_config_, sizeof(STDPRuleConfig));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(STDPRuleConfig);
    
    err = cudaMalloc(&d_neurogenesis_controller_, sizeof(NeurogenesisController));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(NeurogenesisController);
    
    err = cudaMalloc(&d_synaptogenesis_controller_, sizeof(SynaptogenesisController));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(SynaptogenesisController);
    
    err = cudaMalloc(&d_pruning_controller_, sizeof(PruningController));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(PruningController);
    
    err = cudaMalloc(&d_coordination_controller_, sizeof(CoordinationController));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(CoordinationController);
    
    // Environmental interface
    err = cudaMalloc(&d_global_reward_signal_, sizeof(float));
    if (err != cudaSuccess) return false;
    total_memory += sizeof(float);
    
    err = cudaMalloc(&d_environmental_features_, 64 * sizeof(float));
    if (err != cudaSuccess) return false;
    total_memory += 64 * sizeof(float);
    
    err = cudaMalloc(&d_global_neuromodulators_, 4 * sizeof(float));
    if (err != cudaSuccess) return false;
    total_memory += 4 * sizeof(float);
    
    // Performance monitoring
    err = cudaMalloc(&d_performance_metrics_, 32 * sizeof(float));
    if (err != cudaSuccess) return false;
    total_memory += 32 * sizeof(float);
    
    err = cudaMalloc(&d_active_element_counts_, 4 * sizeof(int));
    if (err != cudaSuccess) return false;
    total_memory += 4 * sizeof(int);
    
    peak_memory_usage_mb_ = total_memory / (1024.0f * 1024.0f);
    printf("Total device memory allocated: %.2f MB\n", peak_memory_usage_mb_);
    
    return true;
}

void DynamicNeuralNetwork::deallocateDeviceMemory() {
    // Free all device memory
    if (d_neurons_) cudaFree(d_neurons_);
    if (d_synapses_) cudaFree(d_synapses_);
    if (d_rng_states_) cudaFree(d_rng_states_);
    if (d_plasticity_states_) cudaFree(d_plasticity_states_);
    if (d_dopamine_neurons_) cudaFree(d_dopamine_neurons_);
    if (d_value_functions_) cudaFree(d_value_functions_);
    if (d_actor_critic_states_) cudaFree(d_actor_critic_states_);
    if (d_curiosity_systems_) cudaFree(d_curiosity_systems_);
    if (d_neural_progenitors_) cudaFree(d_neural_progenitors_);
    if (d_developmental_trajectories_) cudaFree(d_developmental_trajectories_);
    if (d_synaptic_progenitors_) cudaFree(d_synaptic_progenitors_);
    if (d_synaptic_competition_) cudaFree(d_synaptic_competition_);
    if (d_neuron_assessments_) cudaFree(d_neuron_assessments_);
    if (d_synapse_assessments_) cudaFree(d_synapse_assessments_);
    if (d_neuron_competition_) cudaFree(d_neuron_competition_);
    if (d_synapse_competition_) cudaFree(d_synapse_competition_);
    if (d_neural_homeostasis_) cudaFree(d_neural_homeostasis_);
    if (d_synaptic_homeostasis_) cudaFree(d_synaptic_homeostasis_);
    if (d_network_homeostasis_) cudaFree(d_network_homeostasis_);
    if (d_stdp_config_) cudaFree(d_stdp_config_);
    if (d_neurogenesis_controller_) cudaFree(d_neurogenesis_controller_);
    if (d_synaptogenesis_controller_) cudaFree(d_synaptogenesis_controller_);
    if (d_pruning_controller_) cudaFree(d_pruning_controller_);
    if (d_coordination_controller_) cudaFree(d_coordination_controller_);
    if (d_global_reward_signal_) cudaFree(d_global_reward_signal_);
    if (d_environmental_features_) cudaFree(d_environmental_features_);
    if (d_global_neuromodulators_) cudaFree(d_global_neuromodulators_);
    if (d_performance_metrics_) cudaFree(d_performance_metrics_);
    if (d_active_element_counts_) cudaFree(d_active_element_counts_);
}

#endif // PHASE3_INTEGRATION_FRAMEWORK_H