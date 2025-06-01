#!/bin/bash

# Simple CUDA test script
# Run this in your CUDA-enabled terminal to verify setup

echo "=== CUDA Setup Verification ==="

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found in PATH"
    echo "Please install CUDA toolkit and add nvcc to PATH"
    exit 1
fi

echo "✅ nvcc found: $(which nvcc)"
echo "CUDA Version: $(nvcc --version | grep release)"

# Test simple CUDA compilation
echo ""
echo "Testing simple CUDA compilation..."
nvcc -o test_cuda_simple test_cuda_simple.cu

if [ $? -eq 0 ]; then
    echo "✅ Simple CUDA compilation successful"
    echo "Running test..."
    ./test_cuda_simple
else
    echo "❌ Simple CUDA compilation failed"
    exit 1
fi

echo ""
echo "Testing NetworkCUDA.cu compilation..."

# Test just the NetworkCUDA.cu file compilation
nvcc -c src/cuda/NetworkCUDA.cu -I src/cuda -o test_network.o 2>&1

if [ $? -eq 0 ]; then
    echo "✅ NetworkCUDA.cu compiled successfully"
    rm -f test_network.o
else
    echo "❌ NetworkCUDA.cu compilation failed"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "Testing STDP kernel compilation..."

# Test STDP kernel compilation
nvcc -c src/cuda/STDPKernel.cu -I src/cuda -o test_stdp.o 2>&1

if [ $? -eq 0 ]; then
    echo "✅ STDPKernel.cu compiled successfully"
    rm -f test_stdp.o
else
    echo "❌ STDPKernel.cu compilation failed"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "=== CUDA Setup Verification Complete ==="
echo "Your CUDA environment is ready!"
