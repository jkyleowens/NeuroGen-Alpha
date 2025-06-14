#!/bin/bash

# Build script for NeuroGen-Alpha with CUDA and clang++
# Run this script outside of VS Code where clang++ is available

echo "==> Starting CUDA build with clang++..."

# Check if clang++ is available
if ! command -v clang++ &> /dev/null; then
    echo "ERROR: clang++ not found in PATH"
    exit 1
fi

# Check if nvcc is available
if ! command -v /opt/cuda/bin/nvcc &> /dev/null; then
    echo "ERROR: nvcc not found at /opt/cuda/bin/nvcc"
    exit 1
fi

echo "==> Compilers found:"
echo "    clang++: $(which clang++)"
echo "    clang++ version: $(clang++ --version | head -1)"
echo "    nvcc: /opt/cuda/bin/nvcc"
echo "    nvcc version: $(/opt/cuda/bin/nvcc --version | grep release)"

echo ""
echo "==> Building project..."

# Clean previous build
make clean

# Build with verbose output
make all 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo ""
    echo "==> Build successful! Executable created: ./neurogen_trader"
    echo "==> Build log saved to: build.log"
else
    echo ""
    echo "==> Build failed. Check build.log for details."
    echo "==> Last 20 lines of build log:"
    tail -20 build.log
    exit 1
fi
