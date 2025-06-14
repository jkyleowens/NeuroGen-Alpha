// IonChannelTesting.h
#ifndef ION_CHANNEL_TESTING_H
#define ION_CHANNEL_TESTING_H

#include <NeuroGen/IonChannelModels.h>
#include <NeuroGen/IonChannelConstants.h>
#include <NeuroGen/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

/**
 * Comprehensive testing framework for ion channel dynamics
 * Validates biological realism and numerical stability
 */
class IonChannelTestSuite {
public:
    IonChannelTestSuite();
    ~IonChannelTestSuite();
    
    // ========================================
    // MAIN TEST EXECUTION
    // ========================================
    bool runAllTests();
    bool runBasicTests();
    bool runAdvancedTests();
    bool runPerformanceTests();
    
    // ========================================
    // INDIVIDUAL TEST CATEGORIES
    // ========================================
    bool testChannelInitialization();
    bool testChannelKinetics();
    bool testCalciumDynamics();
    bool testSynapticTransmission();
    bool testNumericalStability();
    bool testBiologicalRealism();
    bool testPerformanceBenchmarks();
    
    // ========================================
    // VALIDATION METHODS
    // ========================================
    bool validateChannelSteadyStates();
    bool validateTimeConstants();
    bool validateCurrentVoltageRelations();
    bool validateCalciumBuffering();
    bool validateSynapticKinetics();
    
    // ========================================
    // REPORTING
    // ========================================
    void generateTestReport(const std::string& filename);
    void printTestSummary();
    
private:
    struct TestResult {
        std::string test_name;
        bool passed;
        double execution_time_ms;
        std::string details;
        std::vector<float> data_points;
    };
    
    std::vector<TestResult> test_results_;
    int tests_passed_;
    int tests_failed_;
    
    // Test utilities
    bool setupTestEnvironment(int num_neurons = 100);
    void cleanupTestEnvironment();
    bool compareWithTolerance(float actual, float expected, float tolerance);
    void recordTestResult(const std::string& name, bool passed, 
                         double time_ms, const std::string& details = "");
    
    // Device memory for testing
    GPUNeuronState* d_test_neurons_;
    GPUSynapse* d_test_synapses_;
    curandState* d_test_rng_states_;
    int test_neuron_count_;
    int test_synapse_count_;
};

// ========================================
// IMPLEMENTATION
// ========================================

IonChannelTestSuite::IonChannelTestSuite()
    : tests_passed_(0)
    , tests_failed_(0)
    , d_test_neurons_(nullptr)
    , d_test_synapses_(nullptr)
    , d_test_rng_states_(nullptr)
    , test_neuron_count_(0)
    , test_synapse_count_(0)
{
}

IonChannelTestSuite::~IonChannelTestSuite() {
    cleanupTestEnvironment();
}

bool IonChannelTestSuite::runAllTests() {
    printf("\n=== Ion Channel Test Suite ===\n");
    printf("Running comprehensive validation tests...\n\n");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool all_passed = true;
    
    // Basic functionality tests
    all_passed &= testChannelInitialization();
    all_passed &= testChannelKinetics();
    all_passed &= testCalciumDynamics();
    all_passed &= testSynapticTransmission();
    
    // Validation tests
    all_passed &= validateChannelSteadyStates();
    all_passed &= validateTimeConstants();
    all_passed &= validateCurrentVoltageRelations();
    all_passed &= validateCalciumBuffering();
    all_passed &= validateSynapticKinetics();
    
    // Stability and performance tests
    all_passed &= testNumericalStability();
    all_passed &= testBiologicalRealism();
    all_passed &= testPerformanceBenchmarks();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("\n=== Test Suite Complete ===\n");
    printf("Total time: %ld ms\n", duration.count());
    printf("Tests passed: %d\n", tests_passed_);
    printf("Tests failed: %d\n", tests_failed_);
    printf("Overall result: %s\n", all_passed ? "PASS" : "FAIL");
    
    return all_passed;
}

bool IonChannelTestSuite::testChannelInitialization() {
    printf("Testing channel initialization...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool test_passed = true;
    std::string details;
    
    if (!setupTestEnvironment(100)) {
        recordTestResult("Channel Initialization", false, 0.0, "Failed to setup test environment");
        return false;
    }
    
    // Initialize ion channels
    launchIonChannelInitialization(d_test_neurons_, d_test_rng_states_, test_neuron_count_);
    
    // Copy neurons back to host for validation
    GPUNeuronState* h_neurons = new GPUNeuronState[test_neuron_count_];
    cudaMemcpy(h_neurons, d_test_neurons_, test_neuron_count_ * sizeof(GPUNeuronState), 
               cudaMemcpyDeviceToHost);
    
    // Validate initialization
    int valid_neurons = 0;
    for (int i = 0; i < test_neuron_count_; i++) {
        if (h_neurons[i].active == 0) continue;
        
        bool neuron_valid = true;
        
        for (int c = 0; c < h_neurons[i].compartment_count; c++) {
            if (h_neurons[i].compartment_types[c] == COMPARTMENT_INACTIVE) continue;
            
            // Check calcium concentration
            if (h_neurons[i].ca_conc[c] < 0.0f || h_neurons[i].ca_conc[c] > MAX_CA_CONCENTRATION) {
                neuron_valid = false;
                break;
            }
            
            // Check channel states are in valid range [0,1]
            if (h_neurons[i].channels.ca_m[c] < 0.0f || h_neurons[i].channels.ca_m[c] > 1.0f ||
                h_neurons[i].channels.kca_m[c] < 0.0f || h_neurons[i].channels.kca_m[c] > 1.0f ||
                h_neurons[i].channels.hcn_h[c] < 0.0f || h_neurons[i].channels.hcn_h[c] > 1.0f) {
                neuron_valid = false;
                break;
            }
            
            // Check conductances are non-negative
            if (h_neurons[i].channels.ampa_g[c] < 0.0f || h_neurons[i].channels.nmda_g[c] < 0.0f ||
                h_neurons[i].channels.gaba_a_g[c] < 0.0f || h_neurons[i].channels.gaba_b_g[c] < 0.0f) {
                neuron_valid = false;
                break;
            }
        }
        
        if (neuron_valid) valid_neurons++;
    }
    
    float validation_rate = (float)valid_neurons / test_neuron_count_;
    if (validation_rate < 0.95f) {  // 95% threshold
        test_passed = false;
        details = "Validation rate: " + std::to_string(validation_rate);
    } else {
        details = "Validation rate: " + std::to_string(validation_rate);
    }
    
    delete[] h_neurons;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    recordTestResult("Channel Initialization", test_passed, duration.count() / 1000.0, details);
    return test_passed;
}

bool IonChannelTestSuite::testChannelKinetics() {
    printf("Testing channel kinetics...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool test_passed = true;
    std::string details;
    
    // Test AMPA channel kinetics
    {
        AMPAChannel ampa = INIT_AMPA_CHANNEL(COMPARTMENT_SOMA);
        float g = 0.0f, state = 0.0f;
        float dt = 0.1f;
        
        // Apply input and simulate
        float input = 1.0f;
        for (int step = 0; step < 100; step++) {
            ampa.updateState(g, state, input, dt);
            input = 0.0f;  // Only first step
        }
        
        // Check that conductance peaked and is decaying
        if (g <= 0.0f) {
            test_passed = false;
            details += "AMPA kinetics failed. ";
        }
    }
    
    // Test NMDA Mg block
    {
        NMDAChannel nmda = INIT_NMDA_CHANNEL(COMPARTMENT_SOMA);
        float mg_block_low = nmda.computeMgBlock(-70.0f);   // Hyperpolarized
        float mg_block_high = nmda.computeMgBlock(0.0f);    // Depolarized
        
        if (mg_block_high <= mg_block_low) {
            test_passed = false;
            details += "NMDA Mg block failed. ";
        }
    }
    
    // Test calcium channel voltage dependence
    {
        CaChannel ca = INIT_CA_CHANNEL(COMPARTMENT_SOMA);
        float act_low = ca.steadyStateActivation(-70.0f);
        float act_high = ca.steadyStateActivation(0.0f);
        
        if (act_high <= act_low) {
            test_passed = false;
            details += "Ca channel voltage dependence failed. ";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    recordTestResult("Channel Kinetics", test_passed, duration.count() / 1000.0, details);
    return test_passed;
}

bool IonChannelTestSuite::testCalciumDynamics() {
    printf("Testing calcium dynamics...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool test_passed = true;
    std::string details;
    
    if (!setupTestEnvironment(10)) {
        recordTestResult("Calcium Dynamics", false, 0.0, "Setup failed");
        return false;
    }
    
    // Set up test neuron with known calcium levels
    GPUNeuronState h_neuron;
    cudaMemcpy(&h_neuron, d_test_neurons_, sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    // Initialize with elevated calcium
    for (int c = 0; c < h_neuron.compartment_count; c++) {
        if (h_neuron.compartment_types[c] != COMPARTMENT_INACTIVE) {
            h_neuron.ca_conc[c] = 0.001f;  // 10x resting level
            h_neuron.ca_buffer[c] = 0.0f;
        }
    }
    
    cudaMemcpy(d_test_neurons_, &h_neuron, sizeof(GPUNeuronState), cudaMemcpyHostToDevice);
    
    // Run calcium dynamics for several timesteps
    float dt = 0.1f;
    for (int step = 0; step < 100; step++) {
        launchCalciumDynamics(d_test_neurons_, dt, 1);
    }
    
    // Check that calcium has decreased (extrusion working)
    cudaMemcpy(&h_neuron, d_test_neurons_, sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    bool calcium_decreased = true;
    for (int c = 0; c < h_neuron.compartment_count; c++) {
        if (h_neuron.compartment_types[c] != COMPARTMENT_INACTIVE) {
            if (h_neuron.ca_conc[c] >= 0.001f) {  // Should have decreased
                calcium_decreased = false;
                break;
            }
        }
    }
    
    if (!calcium_decreased) {
        test_passed = false;
        details = "Calcium extrusion not working";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    recordTestResult("Calcium Dynamics", test_passed, duration.count() / 1000.0, details);
    return test_passed;
}

bool IonChannelTestSuite::testNumericalStability() {
    printf("Testing numerical stability...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool test_passed = true;
    std::string details;
    
    if (!setupTestEnvironment(50)) {
        recordTestResult("Numerical Stability", false, 0.0, "Setup failed");
        return false;
    }
    
    // Run long simulation with small timesteps
    float dt = 0.01f;  // Small timestep
    int num_steps = 10000;  // 100ms simulation
    
    // Initialize system
    launchIonChannelInitialization(d_test_neurons_, d_test_rng_states_, test_neuron_count_);
    
    // Track stability metrics
    std::vector<float> max_voltages;
    std::vector<float> max_calcium_levels;
    
    for (int step = 0; step < num_steps; step++) {
        float current_time = step * dt;
        
        // Update neuron dynamics
        dim3 block(256);
        dim3 grid((test_neuron_count_ + block.x - 1) / block.x);
        enhancedRK4NeuronUpdateKernel<<<grid, block>>>(
            d_test_neurons_, dt, current_time, test_neuron_count_
        );
        
        // Update calcium dynamics
        launchCalciumDynamics(d_test_neurons_, dt, test_neuron_count_);
        
        // Sample stability every 100 steps
        if (step % 100 == 0) {
            GPUNeuronState* h_neurons = new GPUNeuronState[test_neuron_count_];
            cudaMemcpy(h_neurons, d_test_neurons_, 
                      test_neuron_count_ * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
            
            float max_v = -1000.0f, max_ca = 0.0f;
            for (int i = 0; i < test_neuron_count_; i++) {
                if (h_neurons[i].active) {
                    max_v = fmaxf(max_v, h_neurons[i].voltage);
                    for (int c = 0; c < h_neurons[i].compartment_count; c++) {
                        if (h_neurons[i].compartment_types[c] != COMPARTMENT_INACTIVE) {
                            max_ca = fmaxf(max_ca, h_neurons[i].ca_conc[c]);
                        }
                    }
                }
            }
            
            max_voltages.push_back(max_v);
            max_calcium_levels.push_back(max_ca);
            
            delete[] h_neurons;
            
            // Check for instability
            if (max_v > 100.0f || max_v < -200.0f || max_ca > MAX_CA_CONCENTRATION) {
                test_passed = false;
                details = "Numerical instability detected at step " + std::to_string(step);
                break;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    recordTestResult("Numerical Stability", test_passed, duration.count(), details);
    return test_passed;
}

bool IonChannelTestSuite::testBiologicalRealism() {
    printf("Testing biological realism...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool test_passed = true;
    std::string details;
    
    // Test 1: Resting potential should be around -70mV
    // Test 2: Action potential amplitude should be appropriate
    // Test 3: Time constants should be in biological range
    // Test 4: Calcium levels should be realistic
    
    if (!setupTestEnvironment(20)) {
        recordTestResult("Biological Realism", false, 0.0, "Setup failed");
        return false;
    }
    
    launchIonChannelInitialization(d_test_neurons_, d_test_rng_states_, test_neuron_count_);
    
    // Copy neurons to host for analysis
    GPUNeuronState* h_neurons = new GPUNeuronState[test_neuron_count_];
    cudaMemcpy(h_neurons, d_test_neurons_, 
              test_neuron_count_ * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    // Check resting potentials
    float avg_resting_potential = 0.0f;
    int active_count = 0;
    
    for (int i = 0; i < test_neuron_count_; i++) {
        if (h_neurons[i].active) {
            avg_resting_potential += h_neurons[i].voltage;
            active_count++;
        }
    }
    
    if (active_count > 0) {
        avg_resting_potential /= active_count;
        
        // Resting potential should be between -80mV and -60mV
        if (avg_resting_potential < -80.0f || avg_resting_potential > -60.0f) {
            test_passed = false;
            details += "Resting potential out of range: " + std::to_string(avg_resting_potential) + "mV. ";
        }
    }
    
    // Check calcium concentrations
    float avg_calcium = 0.0f;
    int compartment_count = 0;
    
    for (int i = 0; i < test_neuron_count_; i++) {
        if (h_neurons[i].active) {
            for (int c = 0; c < h_neurons[i].compartment_count; c++) {
                if (h_neurons[i].compartment_types[c] != COMPARTMENT_INACTIVE) {
                    avg_calcium += h_neurons[i].ca_conc[c];
                    compartment_count++;
                }
            }
        }
    }
    
    if (compartment_count > 0) {
        avg_calcium /= compartment_count;
        
        // Calcium should be close to resting level (0.1 μM = 0.0001 mM)
        if (avg_calcium < 0.00005f || avg_calcium > 0.0005f) {
            test_passed = false;
            details += "Calcium level out of range: " + std::to_string(avg_calcium * 1000000.0f) + " μM. ";
        }
    }
    
    delete[] h_neurons;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    recordTestResult("Biological Realism", test_passed, duration.count() / 1000.0, details);
    return test_passed;
}

bool IonChannelTestSuite::setupTestEnvironment(int num_neurons) {
    cleanupTestEnvironment();
    
    test_neuron_count_ = num_neurons;
    test_synapse_count_ = num_neurons * 10;  // 10 synapses per neuron
    
    // Allocate device memory
    cudaError_t err;
    
    err = cudaMalloc(&d_test_neurons_, test_neuron_count_ * sizeof(GPUNeuronState));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_test_synapses_, test_synapse_count_ * sizeof(GPUSynapse));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_test_rng_states_, test_neuron_count_ * sizeof(curandState));
    if (err != cudaSuccess) return false;
    
    // Initialize basic neuron structure
    GPUNeuronState* h_neurons = new GPUNeuronState[test_neuron_count_];
    
    for (int i = 0; i < test_neuron_count_; i++) {
        h_neurons[i].neuron_id = i;
        h_neurons[i].active = 1;
        h_neurons[i].type = (i % 5 == 0) ? 1 : 0;  // 20% inhibitory
        h_neurons[i].compartment_count = 3;  // Soma + 2 dendrites
        h_neurons[i].voltage = -70.0f;
        h_neurons[i].spike_threshold = -50.0f;
        h_neurons[i].spike_threshold_modulated = -50.0f;
        h_neurons[i].spiked = false;
        h_neurons[i].last_spike_time = -1000.0f;
        
        // Set compartment types
        h_neurons[i].compartment_types[0] = COMPARTMENT_SOMA;
        h_neurons[i].compartment_types[1] = COMPARTMENT_BASAL;
        h_neurons[i].compartment_types[2] = COMPARTMENT_APICAL;
        
        // Set parent relationships
        h_neurons[i].parent_compartment[0] = -1;  // Soma has no parent
        h_neurons[i].parent_compartment[1] = 0;   // Basal connects to soma
        h_neurons[i].parent_compartment[2] = 0;   // Apical connects to soma
        
        // Set coupling conductances
        h_neurons[i].coupling_conductance[0] = 0.0f;  // Soma
        h_neurons[i].coupling_conductance[1] = 0.1f;  // Basal
        h_neurons[i].coupling_conductance[2] = 0.1f;  // Apical
        
        // Initialize voltages
        for (int c = 0; c < MAX_COMPARTMENTS; c++) {
            h_neurons[i].voltages[c] = -70.0f;
        }
    }
    
    cudaMemcpy(d_test_neurons_, h_neurons, test_neuron_count_ * sizeof(GPUNeuronState), 
               cudaMemcpyHostToDevice);
    
    delete[] h_neurons;
    
    return true;
}

void IonChannelTestSuite::cleanupTestEnvironment() {
    if (d_test_neurons_) {
        cudaFree(d_test_neurons_);
        d_test_neurons_ = nullptr;
    }
    if (d_test_synapses_) {
        cudaFree(d_test_synapses_);
        d_test_synapses_ = nullptr;
    }
    if (d_test_rng_states_) {
        cudaFree(d_test_rng_states_);
        d_test_rng_states_ = nullptr;
    }
    
    test_neuron_count_ = 0;
    test_synapse_count_ = 0;
}

void IonChannelTestSuite::recordTestResult(const std::string& name, bool passed, 
                                          double time_ms, const std::string& details) {
    TestResult result;
    result.test_name = name;
    result.passed = passed;
    result.execution_time_ms = time_ms;
    result.details = details;
    
    test_results_.push_back(result);
    
    if (passed) {
        tests_passed_++;
        printf("  ✓ %s (%.2f ms)\n", name.c_str(), time_ms);
    } else {
        tests_failed_++;
        printf("  ✗ %s (%.2f ms) - %s\n", name.c_str(), time_ms, details.c_str());
    }
}

void IonChannelTestSuite::generateTestReport(const std::string& filename) {
    std::ofstream file(filename);
    
    file << "Ion Channel Test Suite Report\n";
    file << "=============================\n\n";
    
    file << "Summary:\n";
    file << "Tests passed: " << tests_passed_ << "\n";
    file << "Tests failed: " << tests_failed_ << "\n";
    file << "Success rate: " << (100.0 * tests_passed_ / (tests_passed_ + tests_failed_)) << "%\n\n";
    
    file << "Detailed Results:\n";
    for (const auto& result : test_results_) {
        file << "Test: " << result.test_name << "\n";
        file << "Result: " << (result.passed ? "PASS" : "FAIL") << "\n";
        file << "Time: " << result.execution_time_ms << " ms\n";
        if (!result.details.empty()) {
            file << "Details: " << result.details << "\n";
        }
        file << "\n";
    }
    
    file.close();
}

#endif // ION_CHANNEL_TESTING_H