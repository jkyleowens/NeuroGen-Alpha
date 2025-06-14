#pragma once
#ifndef NETWORK_CUDA_INTERFACE_H
#define NETWORK_CUDA_INTERFACE_H

#include <vector>
#include <string>
#include <NeuroGen/NetworkConfig.h>

// C++ Interface functions for NetworkCUDA
// These provide a simplified interface for main.cpp to interact with the CUDA neural network

// Initialize the neural network with default trading-optimized configuration
void initializeNetwork();

// Perform forward pass through the network
// input: input features for the network
// reward_signal: current reward signal for learning
// returns: network output values
std::vector<float> forwardCUDA(const std::vector<float>& input, float reward_signal);

// Update synaptic weights based on reward signal
// reward_signal: reward signal for learning algorithm
void updateSynapticWeightsCUDA(float reward_signal);

// Get current neuromodulator levels
// returns: vector containing [dopamine, acetylcholine, serotonin, norepinephrine] levels
std::vector<float> getNeuromodulatorLevels();

// Print current network statistics
void printNetworkStats();

// Clean up and destroy the neural network
void cleanupNetwork();

// Advanced configuration functions (optional)
void setNetworkConfig(const NetworkConfig& config);
NetworkConfig getNetworkConfig();
void resetNetwork();
void saveNetworkState(const std::string& filename);
void loadNetworkState(const std::string& filename);

#endif // NETWORK_CUDA_INTERFACE_H
