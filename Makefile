# =============================================================================
# Makefile for NeuroGen-Alpha Project
#
# This Makefile handles the compilation of both C++ and CUDA source files,
# linking them into a final executable for the neural trading agent.
#
# Targets:
#   all           - (Default) Compiles the entire project.
#   install_headers - Copies all source headers to the include/ directory.
#   clean         - Removes all compiled object files and the final executable.
#
# =============================================================================

# --- Compiler and Tool Definitions ---
CXX := g++
NVCC := nvcc
EXEC := neurogen_trader

# --- Directories ---
SRC_DIR := src
OBJ_DIR := obj
INCLUDE_DIR := include
CUDA_SRC_DIR := $(SRC_DIR)/cuda

# --- Compiler Flags ---
# Adjust -arch=sm_XX to match your GPU's compute capability.
# sm_75 (Turing), sm_86 (Ampere), sm_90 (Ada Lovelace) are common.
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -Xcompiler -Wall,-Wno-unused-function
CXXFLAGS := -O3 -std=c++17 -Wall -g
LDFLAGS := -L/usr/local/cuda/lib64
INCLUDES := -I./$(INCLUDE_DIR)
LIBS := -lcudart -lrt

# --- Source Files ---
# Automatically find all .cpp and .cu files in the source directories.
SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU := $(wildcard $(CUDA_SRC_DIR)/*.cu)

# --- Object Files ---
# Generate object file names by replacing src/ with obj/ and extensions with .o
OBJS_CPP := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))
OBJS_CU := $(patsubst $(CUDA_SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

# --- Phony Targets (commands that don't produce a file with the same name) ---
.PHONY: all clean install_headers

# --- Main Build Target ---
all: install_headers $(EXEC)

# --- Linking Rule ---
# This rule links all compiled .o files into the final executable.
$(EXEC): $(OBJS)
	@echo "==> Linking executable: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)
	@echo "==> Build complete. Executable created: $(EXEC)"

# --- CUDA Compilation Rule ---
# This rule compiles .cu files into .o files.
$(OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "==> Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# --- C++ Compilation Rule ---
# This rule compiles .cpp files into .o files.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "==> Compiling C++: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# --- Header Installation Rule ---
# This rule finds all .h and .cuh files and copies them to the include/ directory,
# preserving their subdirectory structure (e.g., cuda/).
install_headers:
	@echo "==> Installing headers..."
	@mkdir -p $(INCLUDE_DIR)/NeuroGen/cuda
	@find $(SRC_DIR) -name "*.h" -not -path "$(CUDA_SRC_DIR)/*" -exec cp --parents {} $(INCLUDE_DIR)/NeuroGen/ \;
	@find $(CUDA_SRC_DIR) -name "*.h" -exec cp --parents {} $(INCLUDE_DIR)/NeuroGen/ \;
	@find $(CUDA_SRC_DIR) -name "*.cuh" -exec cp --parents {} $(INCLUDE_DIR)/NeuroGen/ \;
	@echo "==> Headers installed in $(INCLUDE_DIR)/NeuroGen"


# --- Cleanup Rule ---
# This rule removes the executable and all object files.
clean:
	@echo "==> Cleaning up build files..."
	rm -f $(EXEC)
	rm -rf $(OBJ_DIR)
	@echo "==> Cleanup complete."