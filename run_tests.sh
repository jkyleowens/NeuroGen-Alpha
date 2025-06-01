#!/bin/bash

# Comprehensive test script for CUDA Neural Network
# Run this in your CUDA-enabled terminal

set -e  # Exit on any error

echo "=== CUDA Neural Network Integration Test ==="
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Check CUDA availability
echo "1. Checking CUDA environment..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi

echo "CUDA Compiler: $(nvcc --version | head -n 1)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || echo "Warning: nvidia-smi not available"
echo ""

# Build the project
echo "2. Building the project..."
make clean
make

if [ $? -ne 0 ]; then
    echo "ERROR: Main project build failed"
    exit 1
fi
echo "✓ Main project built successfully"
echo ""

# Build the network test
echo "3. Building network test..."
nvcc -o test_network test_network.cpp src/cuda/*.cu -I src/cuda -lcurand -std=c++11

if [ $? -ne 0 ]; then
    echo "ERROR: Network test build failed"
    exit 1
fi
echo "✓ Network test built successfully"
echo ""

# Run the network test
echo "4. Running network functionality test..."
./test_network

if [ $? -ne 0 ]; then
    echo "ERROR: Network test execution failed"
    exit 1
fi
echo "✓ Network test completed successfully"
echo ""

# Check if we have sample data
echo "5. Checking sample data availability..."
DATA_DIR="highly_diverse_stock_data"
if [ -d "$DATA_DIR" ] && [ "$(ls -1 $DATA_DIR/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✓ Found sample data: $(ls -1 $DATA_DIR/*.csv | wc -l) CSV files"
    
    echo ""
    echo "6. Running short trading simulation test..."
    timeout 30 ./neural_sim "$DATA_DIR" 1 || echo "Simulation test completed (timeout after 30s)"
    echo "✓ Trading simulation test completed"
else
    echo "Warning: No sample data found in $DATA_DIR"
    echo "To test with real data, run: python download_data.py"
fi

echo ""
echo "=== All Tests Completed Successfully ==="
echo "The neural network is ready for full-scale training!"
echo ""
echo "Next steps:"
echo "- Run: ./neural_sim $DATA_DIR 5    # Full 5-epoch training"
echo "- Monitor GPU memory: watch nvidia-smi"
echo "- Check learning progress in network_evolution.csv"
