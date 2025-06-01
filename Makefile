# Practical Makefile for CUDA Neural Trading System
# This version only depends on files that exist or can be easily created

# Check for nvcc
NVCC_PATH := $(shell which nvcc)
ifeq ($(NVCC_PATH),)
$(error "nvcc not found in PATH. Please install CUDA toolkit")
endif

# Basic settings
NVCC := $(NVCC_PATH)
CUDA_HOME ?= /usr/local/cuda

# GPU architecture (auto-detect or default)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.' || echo "75")

# Directories
SRCDIR := src/cuda
OBJDIR := build
BINDIR := bin

# Include paths
INCLUDES := -I./include/NeuroGen -I./include/NeuroGen/cuda -I./src/cuda -I$(CUDA_HOME)/include -I./

# Compiler flags
NVCC_FLAGS := -std=c++17 -O3 $(INCLUDES) \
              --expt-relaxed-constexpr \
              -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
              --use_fast_math

# Debug build option
ifdef DEBUG
    NVCC_FLAGS += -g -G -DDEBUG
    BUILD_TYPE := debug
else
    NVCC_FLAGS += -DNDEBUG
    BUILD_TYPE := release
endif

# Linker flags
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcurand -lcudart

# Source files (only include files that exist)
EXISTING_CU_SRCS := \
    $(wildcard $(SRCDIR)/RandomStateInit.cu) \
    $(wildcard $(SRCDIR)/KernelLaunchWrappers.cu) \
    $(wildcard $(SRCDIR)/NeuronUpdateKernel.cu) \
    $(wildcard $(SRCDIR)/NeuronSpikingKernels.cu) \
    $(wildcard $(SRCDIR)/SynapseInputKernel.cu) \
    $(wildcard $(SRCDIR)/STDPKernel.cu)

# Add NetworkCUDA.cu if it exists, otherwise we'll create a simple one
NETWORK_CUDA_SRC := $(wildcard $(SRCDIR)/NetworkCUDA.cu)
ifeq ($(NETWORK_CUDA_SRC),)
    NETWORK_CUDA_SRC := $(SRCDIR)/NetworkCUDA_simple.cu
endif

CU_SRCS := $(EXISTING_CU_SRCS) $(NETWORK_CUDA_SRC)

# C++ sources
EXISTING_CPP_SRCS := \
    $(wildcard main.cpp) \
    $(wildcard src/TopologyGenerator.cpp)

CPP_SRCS := $(EXISTING_CPP_SRCS)

# Object files
CU_OBJS := $(CU_SRCS:%.cu=$(OBJDIR)/%.cu.o)
CPP_OBJS := $(CPP_SRCS:%.cpp=$(OBJDIR)/%.cpp.o)
ALL_OBJS := $(CU_OBJS) $(CPP_OBJS)

# Target
TARGET := $(BINDIR)/neural_trading_sim

# Default target
all: setup $(TARGET)

# Setup target to create necessary files and directories
setup:
	@echo "=== Setting Up Build Environment ==="
	@echo "CUDA Toolkit: $(CUDA_HOME)"
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "Build Type: $(BUILD_TYPE)"
	@mkdir -p $(OBJDIR) $(BINDIR) $(SRCDIR)
	@$(MAKE) -s check-files

# Check if required files exist and create simple versions if needed
check-files:
	@echo "Checking required files..."
	@if [ ! -f "$(SRCDIR)/NetworkCUDA.cu" ]; then \
		echo "Creating simple NetworkCUDA.cu..."; \
		$(MAKE) -s create-simple-network; \
	fi
	@if [ ! -f "main.cpp" ]; then \
		echo "Creating simple main.cpp..."; \
		$(MAKE) -s create-simple-main; \
	fi

# Create a minimal NetworkCUDA.cu that will compile
create-simple-network:
	@echo '#include <vector>' > $(SRCDIR)/NetworkCUDA.cu
	@echo '#include <iostream>' >> $(SRCDIR)/NetworkCUDA.cu
	@echo '#include <cuda_runtime.h>' >> $(SRCDIR)/NetworkCUDA.cu
	@echo 'void initializeNetwork() { std::cout << "Network initialized\\n"; }' >> $(SRCDIR)/NetworkCUDA.cu
	@echo 'std::vector<float> forwardCUDA(const std::vector<float>& input, float reward) {' >> $(SRCDIR)/NetworkCUDA.cu
	@echo '    return {0.33f, 0.33f, 0.34f}; // Simple placeholder' >> $(SRCDIR)/NetworkCUDA.cu
	@echo '}' >> $(SRCDIR)/NetworkCUDA.cu
	@echo 'void updateSynapticWeightsCUDA(float reward) { /* placeholder */ }' >> $(SRCDIR)/NetworkCUDA.cu
	@echo 'void cleanupNetwork() { std::cout << "Network cleaned up\\n"; }' >> $(SRCDIR)/NetworkCUDA.cu

# Create a minimal main.cpp that will compile
create-simple-main:
	@echo '#include <iostream>' > main.cpp
	@echo '#include <vector>' >> main.cpp
	@echo 'void initializeNetwork();' >> main.cpp
	@echo 'std::vector<float> forwardCUDA(const std::vector<float>&, float);' >> main.cpp
	@echo 'void updateSynapticWeightsCUDA(float);' >> main.cpp
	@echo 'void cleanupNetwork();' >> main.cpp
	@echo 'int main() {' >> main.cpp
	@echo '    std::cout << "Simple Neural Trading System\\n";' >> main.cpp
	@echo '    initializeNetwork();' >> main.cpp
	@echo '    auto result = forwardCUDA({1.0f, 2.0f, 3.0f}, 0.5f);' >> main.cpp
	@echo '    std::cout << "Output: " << result[0] << " " << result[1] << " " << result[2] << "\\n";' >> main.cpp
	@echo '    updateSynapticWeightsCUDA(0.1f);' >> main.cpp
	@echo '    cleanupNetwork();' >> main.cpp
	@echo '    return 0;' >> main.cpp
	@echo '}' >> main.cpp

# Build target
$(TARGET): $(ALL_OBJS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete!"

# Compile CUDA files
$(OBJDIR)/%.cu.o: %.cu
	@echo "Compiling CUDA: $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Compile C++ files  
$(OBJDIR)/%.cpp.o: %.cpp
	@echo "Compiling C++: $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean
clean:
	@echo "Cleaning..."
	rm -rf $(OBJDIR) $(BINDIR)

# Deep clean (remove generated files too)
distclean: clean
	rm -f $(SRCDIR)/NetworkCUDA_simple.cu
	@if [ -f "main.cpp" ] && [ "$$(head -1 main.cpp)" = "#include <iostream>" ]; then \
		echo "Removing generated main.cpp"; \
		rm -f main.cpp; \
	fi

# Install required header files from artifacts
install-headers:
	@echo "To install the full enhanced system, please create these files:"
	@echo "1. src/cuda/NetworkCUDA.cuh - Enhanced network interface"
	@echo "2. src/cuda/CudaUtils.cuh - CUDA utility functions"  
	@echo "3. src/cuda/NetworkConfig.h - Network configuration"
	@echo "4. main.cpp - Enhanced trading simulation"
	@echo "Then run: make clean all"

# Run the built program
run: $(TARGET)
	@echo "Running neural trading simulation..."
	./$(TARGET)

# Check CUDA environment
check-cuda:
	@echo "=== CUDA Environment ==="
	@which nvcc && nvcc --version || echo "NVCC not found"
	@which nvidia-smi && nvidia-smi -L || echo "nvidia-smi not found"
	@echo "CUDA_HOME: $(CUDA_HOME)"

# Help
help:
	@echo "Neural Trading System Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the system (creates simple files if needed)"
	@echo "  setup       - Set up build environment and check files"
	@echo "  clean       - Remove build artifacts"
	@echo "  distclean   - Remove all generated files"
	@echo "  run         - Build and run the program"
	@echo "  check-cuda  - Check CUDA installation"
	@echo "  install-headers - Show how to install full enhanced system"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1     - Build with debug symbols"
	@echo ""
	@echo "The Makefile will automatically create minimal versions of missing files"
	@echo "For the full enhanced system, use 'make install-headers' for instructions"

.PHONY: all setup clean distclean run check-cuda help install-headers check-files create-simple-network create-simple-main

# Don't delete intermediate files
.PRECIOUS: $(OBJDIR)/%.cu.o $(OBJDIR)/%.cpp.o