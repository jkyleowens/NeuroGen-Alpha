# =============================================================================
# Makefile for NeuroGen-Alpha Project (v5 - Corrected Clean Rule)
#
# This Makefile handles the compilation of both C++ and CUDA source files.
# It now dynamically locates the CUDA toolkit and properly cleans all generated files.
#
# =============================================================================

# --- Compiler and Tool Definitions ---
CXX := clang++
NVCC := /opt/cuda/bin/nvcc
EXEC := neurogen_trader

# --- Directories and Paths ---
SRC_DIR := src
OBJ_DIR := obj
INCLUDE_DIR := include
CUDA_SRC_DIR := $(SRC_DIR)/cuda

# Define the CUDA installation path. The ?= allows overriding from the command line.
CUDA_HOME ?= /opt/cuda

# --- Compiler Flags ---
# Adjust -arch=sm_XX to match your GPU's compute capability.
# sm_75 (Turing), sm_86 (Ampere), sm_90 (Ada Lovelace) are common.
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -ccbin=clang++ --expt-relaxed-constexpr -Xcompiler -Wall,-Wno-unused-function,-Wno-unknown-pragmas
CXXFLAGS := -O3 -std=c++17 -Wall -g

# --- Linker Flags and Libraries ---
LDFLAGS := -L$(CUDA_HOME)/lib64
LIBS := -lcudart -lrt

# --- Include Paths ---
# Add the CUDA toolkit's include directory to the standard include path.
INCLUDES := -I./$(INCLUDE_DIR) -I$(CUDA_HOME)/include

# --- Source & Object File Definitions ---
SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU := $(wildcard $(CUDA_SRC_DIR)/*.cu)
OBJS_CPP := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))
OBJS_CU := $(patsubst $(CUDA_SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

# --- Phony Targets (commands that don't produce a file with the same name) ---
.PHONY: all clean install_headers

# --- Main Build Target ---
all: install_headers $(EXEC)

# --- Linking Rule ---
$(EXEC): $(OBJS)
	@echo "==> Linking executable: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)
	@echo "==> Build complete. Executable created: $(EXEC)"

# --- CUDA Compilation Rule ---
$(OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "==> Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# --- C++ Compilation Rule ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "==> Compiling C++: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# --- Header Installation Rule ---
install_headers:
	@echo "==> Installing headers..."
	@mkdir -p $(INCLUDE_DIR)/NeuroGen/cuda
	@find $(SRC_DIR) -maxdepth 1 -name "*.h" -exec cp {} $(INCLUDE_DIR)/NeuroGen/ \;
	@find $(CUDA_SRC_DIR) -type f \( -name "*.h" -o -name "*.cuh" \) -exec cp {} $(INCLUDE_DIR)/NeuroGen/cuda/ \;
	@echo "==> Headers installed in $(INCLUDE_DIR)/NeuroGen"

# --- Cleanup Rule (FIXED) ---
# This rule now removes the executable, all object files, and the generated include directory.
clean:
	@echo "==> Cleaning up build files..."
	rm -f $(EXEC)
	rm -rf $(OBJ_DIR)
	rm -rf $(INCLUDE_DIR)
	@echo "==> Cleanup complete."