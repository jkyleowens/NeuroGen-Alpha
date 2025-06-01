#!/bin/bash
# CUDA Neural Network Compilation Readiness Check
echo "=== NeuroGen Alpha Compilation Readiness Check ==="
echo "Date: $(date)"
echo ""

# Counters
PASSED=0
FAILED=0

check_pass() {
    echo "✓ PASS: $1"
    ((PASSED++))
}

check_fail() {
    echo "✗ FAIL: $1"
    ((FAILED++))
}

echo "1. Directory Structure Check"
echo "----------------------------"

# Check main directories
if [ -d "src/cuda" ]; then
    check_pass "src/cuda directory exists"
else
    check_fail "src/cuda directory missing"
fi

if [ -d "include/NeuroGen" ]; then
    check_pass "include/NeuroGen directory exists"
else
    check_fail "include/NeuroGen directory missing"
fi

if [ -d "include/NeuroGen/cuda" ]; then
    check_pass "include/NeuroGen/cuda directory exists"
else
    check_fail "include/NeuroGen/cuda directory missing"
fi

echo ""
echo "2. Core Source Files Check"
echo "--------------------------"

# Check essential CUDA source files
if [ -f "src/cuda/NetworkCUDA.cu" ]; then
    check_pass "NetworkCUDA.cu exists"
else
    check_fail "NetworkCUDA.cu missing"
fi

if [ -f "src/cuda/STDPKernel.cu" ]; then
    check_pass "STDPKernel.cu exists"
else
    check_fail "STDPKernel.cu missing"
fi

if [ -f "src/cuda/NeuronUpdateKernel.cu" ]; then
    check_pass "NeuronUpdateKernel.cu exists"
else
    check_fail "NeuronUpdateKernel.cu missing"
fi

echo ""
echo "3. Header Files Check"
echo "--------------------"

if [ -f "include/NeuroGen/cuda/NetworkCUDA.cuh" ]; then
    check_pass "NetworkCUDA.cuh exists"
else
    check_fail "NetworkCUDA.cuh missing"
fi

if [ -f "include/NeuroGen/cuda/STDPKernel.cuh" ]; then
    check_pass "STDPKernel.cuh exists"
else
    check_fail "STDPKernel.cuh missing"
fi

if [ -f "include/NeuroGen/NetworkConfig.h" ]; then
    check_pass "NetworkConfig.h exists"
else
    check_fail "NetworkConfig.h missing"
fi

if [ -f "include/NeuroGen/GPUNeuralStructures.h" ]; then
    check_pass "GPUNeuralStructures.h exists"
else
    check_fail "GPUNeuralStructures.h missing"
fi

echo ""
echo "4. Include Path Check"
echo "--------------------"

if [ -f "src/cuda/NetworkCUDA.cu" ]; then
    if grep -q "../../include/NeuroGen" "src/cuda/NetworkCUDA.cu"; then
        check_pass "NetworkCUDA.cu has correct include paths"
    else
        check_fail "NetworkCUDA.cu has incorrect include paths"
    fi
fi

echo ""
echo "5. Function Implementation Check"
echo "-------------------------------"

if [ -f "src/cuda/NetworkCUDA.cu" ]; then
    if grep -q "void initializeNetwork" "src/cuda/NetworkCUDA.cu"; then
        check_pass "initializeNetwork function found"
    else
        check_fail "initializeNetwork function missing"
    fi
    
    if grep -q "forwardCUDA" "src/cuda/NetworkCUDA.cu"; then
        check_pass "forwardCUDA function found"
    else
        check_fail "forwardCUDA function missing"
    fi
    
    if grep -q "namespace NetworkCUDAInternal" "src/cuda/NetworkCUDA.cu"; then
        check_pass "NetworkCUDAInternal namespace found"
    else
        check_fail "NetworkCUDAInternal namespace missing"
    fi
fi

echo ""
echo "6. Makefile Check"
echo "----------------"

if [ -f "Makefile" ]; then
    check_pass "Makefile exists"
    
    if grep -q "include/NeuroGen" "Makefile"; then
        check_pass "Makefile has correct include paths"
    else
        check_fail "Makefile missing include paths"
    fi
else
    check_fail "Makefile missing"
fi

echo ""
echo "=== SUMMARY ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🎉 PROJECT IS READY FOR COMPILATION"
    echo "All essential files and structure are in place."
else
    echo ""
    echo "⚠ PROJECT NEEDS FIXES"
    echo "Please address the failed checks before compilation."
fi
