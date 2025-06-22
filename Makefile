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
CXX := clang++
NVCC := nvcc
LINKER := $(NVCC)

# Directories
# Using dedicated build directories keeps the source tree clean.
SRCDIR := src
CUDA_SRCDIR := $(SRCDIR)/cuda
INCLUDEDIR := include
OBJDIR := obj
BINDIR := bin

# Executable Names
EXEC_MAIN := $(BINDIR)/NeuroGen-Alpha
EXEC_TRADING := $(BINDIR)/autonomous_trading

# Find all C++ and CUDA source files automatically
SRC_CPP := $(wildcard $(SRCDIR)/*.cpp)
SRC_CU := $(wildcard $(CUDA_SRCDIR)/*.cu)

# Generate corresponding object file paths in the OBJDIR
OBJS_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRC_CPP))
OBJS_CU := $(patsubst $(CUDA_SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRC_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

# Autonomous Trading Agent specific source files
TRADING_SRCS := $(SRCDIR)/TechnicalAnalysis.cpp \
                $(SRCDIR)/NeuralNetworkInterface.cpp \
				$(SRCDIR)/NetworkPresets.cpp \
                $(SRCDIR)/Portfolio.cpp \
                $(SRCDIR)/CoinbaseAdvancedTradeApi.cpp \
                $(SRCDIR)/AutonomousTradingAgent.cpp \
                $(SRCDIR)/Simulation.cpp \
                $(SRCDIR)/autonomous_trading_main.cpp

# Trading agent object files (C++ only)
TRADING_OBJS_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(TRADING_SRCS))

# Core neural network object files (excluding trading main and main)
CORE_OBJS := $(filter-out $(OBJDIR)/autonomous_trading_main.o $(OBJDIR)/main.o, $(OBJS))

# Combined trading objects (C++ trading files + all CUDA files for neural network)
TRADING_OBJS := $(TRADING_OBJS_CPP) $(OBJS_CU)

# --- External Dependencies ---
# nlohmann/json header-only library
NLOHMANN_JSON_URL := https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
NLOHMANN_JSON_DIR := $(INCLUDEDIR)/nlohmann
NLOHMANN_JSON_HEADER := $(NLOHMANN_JSON_DIR)/json.hpp

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
# Core libraries: CUDA runtime, CUDA random number generator, C++ filesystem
CORE_LDLIBS := -lcudart -lcurand -lstdc++fs

# Trading agent libraries: HTTP client and JSON parsing
# Note: Using nlohmann/json (header-only) instead of jsoncpp
TRADING_LDLIBS := $(CORE_LDLIBS) -lcurl -lssl -lcrypto -Xcompiler -pthread

# Default libraries for main executable
LDLIBS := $(TRADING_LDLIBS)


# ----------------- #
#      Targets      #
# ----------------- #

# Phony targets do not represent actual files.
.PHONY: all core trading run run-trading clean setup-headers setup-deps

# The default target builds both executables
all: setup-deps setup-headers $(EXEC_MAIN) $(EXEC_TRADING)

# Build just the core neural network executable
core: setup-deps setup-headers $(EXEC_MAIN)

# Build just the trading agent executable  
trading: setup-deps setup-headers $(EXEC_TRADING)

# Setup external dependencies
setup-deps: $(NLOHMANN_JSON_HEADER)

# Download nlohmann/json header if not present
$(NLOHMANN_JSON_HEADER):
	@echo "Setting up nlohmann/json dependency..."
	@mkdir -p $(NLOHMANN_JSON_DIR)
	@if command -v wget >/dev/null 2>&1; then \
		wget -q $(NLOHMANN_JSON_URL) -O $(NLOHMANN_JSON_HEADER); \
	elif command -v curl >/dev/null 2>&1; then \
		curl -s -L $(NLOHMANN_JSON_URL) -o $(NLOHMANN_JSON_HEADER); \
	else \
		echo "Error: Neither wget nor curl found. Please install one of them or manually download nlohmann/json.hpp"; \
		exit 1; \
	fi
	@echo "nlohmann/json dependency setup complete."

# Setup header files in the include directory
setup-headers:
	@echo "Setting up header files..."
	@mkdir -p $(INCLUDEDIR)/NeuroGen/cuda
	@# Copy ALL header files from src root directory
	@if [ -f "src/AdvancedReinforcementLearning.h" ]; then cp src/AdvancedReinforcementLearning.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/AutonomousTradingAgent.h" ]; then cp src/AutonomousTradingAgent.h $(INCLUDEDIR)/NeuroGen/; fi
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
	@if [ -f "src/MarketData.h" ]; then cp src/MarketData.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NeuralNetworkInterface.h" ]; then cp src/NeuralNetworkInterface.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkConfig.h" ]; then cp src/NetworkConfig.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Network.h" ]; then cp src/Network.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkIntegration.h" ]; then cp src/NetworkIntegration.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkPresets.h" ]; then cp src/NetworkPresets.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NetworkUpdateStub.h" ]; then cp src/NetworkUpdateStub.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/NeuralPruningFramework.h" ]; then cp src/NeuralPruningFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/PerformanceOptimization.h" ]; then cp src/PerformanceOptimization.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Phase3IntegrationFramework.h" ]; then cp src/Phase3IntegrationFramework.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Portfolio.h" ]; then cp src/Portfolio.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/PriceTick.h" ]; then cp src/PriceTick.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/Simulation.h" ]; then cp src/Simulation.h $(INCLUDEDIR)/NeuroGen/; fi
	@if [ -f "src/TechnicalAnalysis.h" ]; then cp src/TechnicalAnalysis.h $(INCLUDEDIR)/NeuroGen/; fi
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

# Rule to link the main neural network executable.
# Depends on core object files and the existence of the bin directory.
$(EXEC_MAIN): $(CORE_OBJS) $(OBJDIR)/main.o | $(BINDIR)
	@echo "Linking main neural network executable..."
	$(LINKER) $(CORE_OBJS) $(OBJDIR)/main.o -o $@ $(LDFLAGS) $(TRADING_LDLIBS)
	@echo "Main executable build complete: $(EXEC_MAIN)"

# Rule to link the autonomous trading executable.
# Depends on trading object files and the existence of the bin directory.
$(EXEC_TRADING): $(TRADING_OBJS) | $(BINDIR)
	@echo "Linking autonomous trading executable..."
	$(LINKER) $(TRADING_OBJS) -o $@ $(LDFLAGS) $(TRADING_LDLIBS)
	@echo "Trading executable build complete: $(EXEC_TRADING)"

# Rule to compile a C++ source file into an object file.
# Depends on the source file, dependencies, headers being set up, and the existence of the obj directory.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR) setup-deps setup-headers
	@echo "Compiling C++: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Rule to compile a CUDA source file into an object file.
# Depends on the source file, dependencies, headers being set up, and the existence of the obj directory.
$(OBJDIR)/%.o: $(CUDA_SRCDIR)/%.cu | $(OBJDIR) setup-deps setup-headers
	@echo "Compiling CUDA: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# Create the binary and object directories if they don't exist.
$(BINDIR) $(OBJDIR):
	mkdir -p $@

# Target to run the main neural network application.
# Depends on 'core' to ensure the program is built first.
run: core
	@echo "--- Running NeuroGen-Alpha Core ---"
	./$(EXEC_MAIN)
	@echo "---      Run complete      ---"

# Target to run the autonomous trading application.
# Depends on 'trading' to ensure the program is built first.
run-trading: trading
	@echo "--- Running Autonomous Trading Agent ---"
	./$(EXEC_TRADING)
	@echo "---      Run complete      ---"

# Convenience targets for trading with specific parameters
run-trading-btc: trading
	@echo "--- Running Trading Agent (BTC/USD, 1h, 100 ticks) ---"
	./$(EXEC_TRADING) --pair BTCUSD --interval 1h --start "2024-01-01 00:00:00" --end "2024-12-31 23:59:59" --cash 10000 --ticks 100

run-trading-eth: trading
	@echo "--- Running Trading Agent (ETH/USD, 1h, 100 ticks) ---"
	./$(EXEC_TRADING) --pair ETHUSD --interval 1h --start "2024-01-01 00:00:00" --end "2024-12-31 23:59:59" --cash 10000 --ticks 100

# Test trading with smaller datasets for faster iteration
run-trading-test: trading
	@echo "--- Running Trading Agent (Test mode: 20 ticks) ---"
	./$(EXEC_TRADING) --pair BTCUSD --interval 1h --cash 1000 --ticks 20

# Save and load trading state examples
run-trading-save: trading
	@echo "--- Running Trading Agent with state saving ---"
	./$(EXEC_TRADING) --pair BTCUSD --interval 1h --start "2024-01-01 00:00:00" --end "2024-06-30 23:59:59" --cash 10000 --save trading_state --ticks 50

run-trading-load: trading
	@echo "--- Running Trading Agent with state loading ---"
	./$(EXEC_TRADING) --load trading_state --ticks 50

# Build debug version of trading agent
trading-debug: CXXFLAGS += -DDEBUG -O0
trading-debug: NVCCFLAGS += -DDEBUG -O0 
trading-debug: trading
	@echo "Debug build complete: $(EXEC_TRADING)"

# Check status and list important files
status:
	@echo "=== NeuroGen-Alpha Build System Status ==="
	@echo "Source files found:"
	@echo "  C++ files: $(words $(SRC_CPP))"
	@echo "  CUDA files: $(words $(SRC_CU))"
	@echo "  Trading C++ files: $(words $(TRADING_SRCS))"
	@echo ""
	@echo "Trading source files:"
	@$(foreach src,$(TRADING_SRCS),echo "  $(src)";)
	@echo ""
	@echo "Dependencies:"
	@if [ -f "$(NLOHMANN_JSON_HEADER)" ]; then echo "  ✓ nlohmann/json found"; else echo "  ✗ nlohmann/json missing"; fi
	@if command -v nvcc >/dev/null 2>&1; then echo "  ✓ NVCC compiler found"; else echo "  ✗ NVCC compiler missing"; fi
	@if command -v g++ >/dev/null 2>&1; then echo "  ✓ G++ compiler found"; else echo "  ✗ G++ compiler missing"; fi
	@if command -v curl >/dev/null 2>&1; then echo "  ✓ curl found"; else echo "  ✗ curl missing"; fi
	@echo ""
	@echo "Build targets available:"
	@echo "  make all          - Build both core and trading executables"
	@echo "  make core         - Build core neural network only"  
	@echo "  make trading      - Build autonomous trading agent"
	@echo "  make trading-debug - Build trading agent with debug flags"
	@echo "  make run-trading-test - Quick test run with 20 ticks"
	@echo "  make clean        - Clean all build artifacts"

# Target to clean up all build artifacts.
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OBJDIR) $(BINDIR)
	rm -rf $(INCLUDEDIR)/NeuroGen
	rm -rf $(NLOHMANN_JSON_DIR)
	@echo "Cleanup complete."

# Additional phony targets for convenience
.PHONY: run-trading-btc run-trading-eth run-trading-test run-trading-save run-trading-load trading-debug status setup-deps