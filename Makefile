# Makefile for NeuroGen Alpha project with updated directory structure
# Check for nvcc in PATH and error out if missing
NVCC_PATH := $(shell which nvcc)
ifeq ($(NVCC_PATH),)
$(error "nvcc not found in PATH. Please install the CUDA toolkit and ensure `nvcc` is in your PATH.")
else
NVCC := $(NVCC_PATH)
endif

# Compiler and CUDA configuration
CUDA_HOME ?= /usr/local/cuda

# Include and library directories
INCLUDE_DIRS := -I./include -I./include/NeuroGen -I./include/NeuroGen/cuda -I$(CUDA_HOME)/include
LIB_DIRS := -L$(CUDA_HOME)/lib64

# Compilation flags
CXXFLAGS := -std=c++17 -O2 $(INCLUDE_DIRS) --expt-relaxed-constexpr -DUSE_CUDA=1
LDFLAGS := $(LIB_DIRS) -lcudart -lcurand

# Directory structure
SRC_DIR := src
SRC_CUDA_DIR := $(SRC_DIR)/cuda
OBJ_DIR := obj
BIN_DIR := bin
TEST_DIR := tests
INCLUDE_DIR := include

# Make sure the directories exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(OBJ_DIR)/cuda)
$(shell mkdir -p $(INCLUDE_DIR)/NeuroGen)
$(shell mkdir -p $(INCLUDE_DIR)/NeuroGen/cuda)

# Source files
CU_SRCS = $(wildcard $(SRC_CUDA_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJS = $(patsubst $(SRC_CUDA_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))

# Target binary
TARGET = $(BIN_DIR)/neural_sim

# Main targets
.PHONY: all clean test setup headers

all: setup headers $(TARGET)

# Setup directory structure and copy headers
setup:
	@mkdir -p $(INCLUDE_DIR)/NeuroGen/cuda
	@echo "Directory structure created"

# Copy headers to proper include directory structure
headers:
	@echo "Setting up header files..."
	@if [ -f "src/cuda/GPUNeuralStructures.h" ]; then cp src/cuda/GPUNeuralStructures.h $(INCLUDE_DIR)/NeuroGen/; fi
	@if [ -f "src/cuda/NetworkCUDA.cuh" ]; then cp src/cuda/NetworkCUDA.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/STDPKernel.cuh" ]; then cp src/cuda/STDPKernel.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronUpdateKernel.cuh" ]; then cp src/cuda/NeuronUpdateKernel.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronSpikingKernels.cuh" ]; then cp src/cuda/NeuronSpikingKernels.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/KernelLaunchWrappers.cuh" ]; then cp src/cuda/KernelLaunchWrappers.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/SynapseInputKernel.cuh" ]; then cp src/cuda/SynapseInputKernel.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/RandomStateInit.cuh" ]; then cp src/cuda/RandomStateInit.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/GridBlockUtils.cuh" ]; then cp src/cuda/GridBlockUtils.cuh $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/CorticalColumn.h" ]; then cp src/cuda/CorticalColumn.h $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/CudaCompatibility.h" ]; then cp src/cuda/CudaCompatibility.h $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/CudaUtils.h" ]; then cp src/cuda/CudaUtils.h $(INCLUDE_DIR)/NeuroGen/cuda/; fi
	@if [ -f "src/NetworkConfig.h" ]; then cp src/NetworkConfig.h $(INCLUDE_DIR)/NeuroGen/; fi
	@if [ -f "src/NetworkPresets.h" ]; then cp src/NetworkPresets.h $(INCLUDE_DIR)/NeuroGen/; fi
	@if [ -f "src/TopologyGenerator.h" ]; then cp src/TopologyGenerator.h $(INCLUDE_DIR)/NeuroGen/; fi
	@if [ -f "src/Network.h" ]; then cp src/Network.h $(INCLUDE_DIR)/NeuroGen/; fi
	@if [ -f "src/Neuron.h" ]; then cp src/Neuron.h $(INCLUDE_DIR)/NeuroGen/; fi
	@echo "Headers copied to include directory"

# Link all object files, including main.cpp
$(TARGET): $(CU_OBJS) $(CPP_OBJS) $(OBJ_DIR)/main.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

# Compile CUDA sources
$(OBJ_DIR)/cuda/%.o: $(SRC_CUDA_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)/cuda
	$(NVCC) $(CXXFLAGS) -dc $< -o $@

# Compile C++ sources  
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Special rule for main.cpp
$(OBJ_DIR)/main.o: main.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Test binary
TEST_TARGET = $(BIN_DIR)/test_network
TEST_SRCS = $(TEST_DIR)/test_network.cpp
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))

test: setup headers $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS) $(CU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Test build complete: $@"

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(INCLUDE_DIR)/*
	rm -f $(TARGET) $(TEST_TARGET)
	@echo "Clean complete"

# Print info for debugging
info:
	@echo "NVCC: $(NVCC)"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "INCLUDE_DIRS: $(INCLUDE_DIRS)"
	@echo "CUDA sources: $(CU_SRCS)"
	@echo "C++ sources: $(CPP_SRCS)"
	@echo "CUDA objects: $(CU_OBJS)"
	@echo "C++ objects: $(CPP_OBJS)"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@$(NVCC) --version
	@echo "CUDA devices:"
	@nvidia-smi -L 2>/dev/null || echo "nvidia-smi not found or no devices"

# Create minimal test to verify compilation
minimal-test:
	@echo "Creating minimal test..."
	@echo '#include <iostream>' > minimal_test.cpp
	@echo '#include <cuda_runtime.h>' >> minimal_test.cpp
	@echo 'int main() { std::cout << "CUDA test passed" << std::endl; return 0; }' >> minimal_test.cpp
	$(NVCC) $(CXXFLAGS) minimal_test.cpp -o $(BIN_DIR)/minimal_test $(LDFLAGS)
	@echo "Minimal test compiled successfully"
	@rm minimal_test.cpp
