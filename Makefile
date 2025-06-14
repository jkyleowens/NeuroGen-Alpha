#
# Makefile for the NeuroGen-Alpha Project
#
# This Makefile handles the compilation of both C++ and CUDA source files,
# linking them into a single executable, and provides run/clean commands.
#

# ----------------- #
#  Configuration    #
# ----------------- #

# Compilers and Linker
# We use nvcc for the final linking stage to simplify inclusion of CUDA libraries.
CXX := g++
NVCC := nvcc
LINKER := $(NVCC)

# Directories
# Using dedicated build directories keeps the source tree clean.
SRCDIR := src
CUDA_SRCDIR := $(SRCDIR)/cuda
INCLUDEDIR := include
OBJDIR := obj
BINDIR := bin

# Executable Name
EXEC := $(BINDIR)/NeuroGen-Alpha

# Find all C++ and CUDA source files automatically
SRC_CPP := $(wildcard $(SRCDIR)/*.cpp)
SRC_CU := $(wildcard $(CUDA_SRCDIR)/*.cu)

# Generate corresponding object file paths in the OBJDIR
OBJS_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRC_CPP))
OBJS_CU := $(patsubst $(CUDA_SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRC_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

# --- Library Paths ---
# Adjust these paths if CUDA or Poco are installed in non-standard locations.
CUDA_PATH ?= /usr/local/cuda
POCO_PATH ?= /usr/local/poco

# --- Compiler and Linker Flags ---

# Include paths for headers (-I)
CPPFLAGS := -I$(INCLUDEDIR) -I$(CUDA_PATH)/include

# C++ specific compiler flags
# Using C++17 standard as requested
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -g

# NVCC specific compiler flags
# Using C++17 standard as requested
# -arch=native compiles for the architecture of the GPU on the build machine.
NVCCFLAGS := -std=c++17 -arch=native -O2 -g --compiler-options "-Wall -Wextra"

# Library paths for linker (-L)
LDFLAGS := -L$(CUDA_PATH)/lib64

# Libraries to link against (-l)
# Note: Linking CUDA runtime, CUDA random number generator, libcurl for HTTP requests, and jsoncpp for JSON parsing
LDLIBS := -lcudart -lcurand -lstdc++fs -lcurl -ljsoncpp


# ----------------- #
#      Targets      #
# ----------------- #

# Phony targets do not represent actual files.
.PHONY: all run clean setup-headers

# The default target, 'all', is the first target in the file.
all: setup-headers $(EXEC)

# Setup header files in the include directory
setup-headers:
	@echo "Setting up header files..."
	@mkdir -p $(INCLUDEDIR)/NeuroGen/cuda
	@# Copy ALL header files from src root directory
	@if [ -f "src/AdvancedReinforcementLearning.h" ]; then cp src/AdvancedReinforcementLearning.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/DataStructures.h" ]; then cp src/DataStructures.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/DynamicNeurogenesisFramework.h" ]; then cp src/DynamicNeurogenesisFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/DynamicSynaptogenesisFramework.h" ]; then cp src/DynamicSynaptogenesisFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/EnhancedLearningSystem.h" ]; then cp src/EnhancedLearningSystem.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/EnhancedSTDPFramework.h" ]; then cp src/EnhancedSTDPFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/GPUCorticalColumnFwd.h" ]; then cp src/GPUCorticalColumnFwd.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/GPUStructuresFwd.h" ]; then cp src/GPUStructuresFwd.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/HomeostaticRegulationSystem.h" ]; then cp src/HomeostaticRegulationSystem.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/IntegratedSimulationLoop.h" ]; then cp src/IntegratedSimulationLoop.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/IonChannelConstants.h" ]; then cp src/IonChannelConstants.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/IonChannelModels.h" ]; then cp src/IonChannelModels.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/IonChannelTesting.h" ]; then cp src/IonChannelTesting.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/LearningRuleConstants.h" ]; then cp src/LearningRuleConstants.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkConfig.h" ]; then cp src/NetworkConfig.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Network.h" ]; then cp src/Network.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkIntegration.h" ]; then cp src/NetworkIntegration.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkPresets.h" ]; then cp src/NetworkPresets.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkUpdateStub.h" ]; then cp src/NetworkUpdateStub.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NeuralPruningFramework.h" ]; then cp src/NeuralPruningFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/PerformanceOptimization.h" ]; then cp src/PerformanceOptimization.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Phase3IntegrationFramework.h" ]; then cp src/Phase3IntegrationFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/TopologyGenerator.h" ]; then cp src/TopologyGenerator.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/TradingAgent.h" ]; then cp src/TradingAgent.h $(INCLUDEDIR)/NeuroGen/; fi
	@# Copy ALL CUDA header files
	@if [ -f "src/cuda/CorticalColumn.h" ]; then cp src/cuda/CorticalColumn.h $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/CudaCompatibility.h" ]; then cp src/cuda/CudaCompatibility.h $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/CudaUtils.h" ]; then cp src/cuda/CudaUtils.h $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/EligibilityAndRewardKernels.cuh" ]; then cp src/cuda/EligibilityAndRewardKernels.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/EligibilityTraceKernel.cuh" ]; then cp src/cuda/EligibilityTraceKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/EnhancedSTDPKernel.cuh" ]; then cp src/cuda/EnhancedSTDPKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/GPUNeuralStructures.h" ]; then cp src/cuda/GPUNeuralStructures.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/cuda/GridBlockUtils.cuh" ]; then cp src/cuda/GridBlockUtils.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/HebbianLearningKernel.cuh" ]; then cp src/cuda/HebbianLearningKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/HomeostaticMechanismsKernel.cuh" ]; then cp src/cuda/HomeostaticMechanismsKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/KernelLaunchWrappers.cuh" ]; then cp src/cuda/KernelLaunchWrappers.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NetworkCUDA.cuh" ]; then cp src/cuda/NetworkCUDA.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NetworkCUDA_Interface.h" ]; then cp src/cuda/NetworkCUDA_Interface.h $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronInitialization.cuh" ]; then cp src/cuda/NeuronInitialization.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronModelConstants.h" ]; then cp src/cuda/NeuronModelConstants.h $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronSpikingKernels.cuh" ]; then cp src/cuda/NeuronSpikingKernels.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/NeuronUpdateKernel.cuh" ]; then cp src/cuda/NeuronUpdateKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/RandomStateInit.cuh" ]; then cp src/cuda/RandomStateInit.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/RewardModulationKernel.cuh" ]; then cp src/cuda/RewardModulationKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/StructuralPlasticityKernels.cuh" ]; then cp src/cuda/StructuralPlasticityKernels.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@if [ -f "src/cuda/SynapseInputKernel.cuh" ]; then cp src/cuda/SynapseInputKernel.cuh $(INCLUDEDIR)/NeuroGen/cuda/; fi
	@echo "All header files copied to include directory"

# Rule to link the final executable.
# Depends on all object files and the existence of the bin directory.
$(EXEC): $(OBJS) | $(BINDIR)
	@echo "Linking executable..."
	$(LINKER) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Executable is at: $(EXEC)"

# Rule to compile a C++ source file into an object file.
# Depends on the source file, headers being set up, and the existence of the obj directory.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR) setup-headers
	@echo "Compiling C++: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Rule to compile a CUDA source file into an object file.
# Depends on the source file, headers being set up, and the existence of the obj directory.
$(OBJDIR)/%.o: $(CUDA_SRCDIR)/%.cu | $(OBJDIR) setup-headers
	@echo "Compiling CUDA: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# Create the binary and object directories if they don't exist.
$(BINDIR) $(OBJDIR):
	mkdir -p $@

# Target to run the compiled application.
# Depends on 'all' to ensure the program is built first.
run: all
	@echo "--- Running NeuroGen-Alpha ---"
	./$(EXEC)
	@echo "---      Run complete      ---"

# Target to clean up all build artifacts.
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OBJDIR) $(BINDIR)
	rm -rf $(INCLUDEDIR)/NeuroGen
	@echo "Cleanup complete."