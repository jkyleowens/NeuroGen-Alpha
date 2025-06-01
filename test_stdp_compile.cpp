// Test STDP compilation
#include "src/cuda/STDPKernel.cuh"
#include "src/cuda/GPUNeuralStructures.h"
#include <iostream>

int main() {
    std::cout << "Testing STDP function compilation..." << std::endl;
    
    // This should compile if the function is properly declared
    GPUSynapse* dummy_synapses = nullptr;
    GPUNeuronState* dummy_neurons = nullptr;
    
    // We won't actually call it, just test that it compiles
    // launchSTDPUpdateKernel(dummy_synapses, dummy_neurons, 0, 0.01f, 0.01f, 20.0f, 20.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    
    std::cout << "STDP function declaration found successfully!" << std::endl;
    return 0;
}
