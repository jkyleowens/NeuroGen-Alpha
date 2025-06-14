#!/bin/bash

echo "Installing NeuroGen-Alpha Dependencies"
echo "====================================="

# Update package manager
echo "Updating package manager..."
sudo apt-get update

# Install basic development tools
echo "Installing development tools..."
sudo apt-get install -y build-essential cmake pkg-config

# Install CUDA development tools (if not already installed)
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please install CUDA Toolkit manually."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
else
    echo "CUDA found: $(nvcc --version | grep release)"
fi

# Install libcurl for HTTP requests
echo "Installing libcurl..."
sudo apt-get install -y libcurl4-openssl-dev

# Install jsoncpp for JSON parsing
echo "Installing jsoncpp..."
sudo apt-get install -y libjsoncpp-dev

# Install additional useful libraries
echo "Installing additional libraries..."
sudo apt-get install -y libssl-dev zlib1g-dev

# Verify installations
echo ""
echo "Verification:"
echo "============="

if pkg-config --exists libcurl; then
    echo "✓ libcurl: $(pkg-config --modversion libcurl)"
else
    echo "✗ libcurl: Not found"
fi

if pkg-config --exists jsoncpp; then
    echo "✓ jsoncpp: $(pkg-config --modversion jsoncpp)"
else
    echo "✗ jsoncpp: Not found"
fi

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA: Available"
else
    echo "✗ CUDA: Not found"
fi

echo ""
echo "Dependencies installation completed!"
echo "You can now build the project with: make clean && make"
