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
INCLUDE_DIRS := -I./include -I$(CUDA_HOME)/include
LIB_DIRS := -L$(CUDA_HOME)/lib64

# Compilation flags
CXXFLAGS := -std=c++17 -O2 $(INCLUDE_DIRS) --expt-relaxed-constexpr -DUSE_CUDA=1
LDFLAGS := $(LIB_DIRS) -lcurand

# Directory structure
SRC_DIR := src
SRC_CUDA_DIR := $(SRC_DIR)/cuda
OBJ_DIR := obj
BIN_DIR := bin
TEST_DIR := tests

# Make sure the directories exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(OBJ_DIR)/cuda)

# Source files
CU_SRCS = $(wildcard $(SRC_CUDA_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJS = $(patsubst $(SRC_CUDA_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))

# Target binary
TARGET = $(BIN_DIR)/neural_sim

# Main targets
.PHONY: all clean test

all: $(TARGET)

# Link all object files
$(TARGET): $(CU_OBJS) $(CPP_OBJS)
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

# Test binary
TEST_TARGET = $(BIN_DIR)/test_network
TEST_SRCS = $(TEST_DIR)/test_network.cpp
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))

test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS) $(CU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Test build complete: $@"

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*
	rm -f $(TARGET) $(TEST_TARGET)
	@echo "Clean complete"

# Print info for debugging
info:
	@echo "CUDA sources: $(CU_SRCS)"
	@echo "C++ sources: $(CPP_SRCS)"
	@echo "CUDA objects: $(CU_OBJS)"
	@echo "C++ objects: $(CPP_OBJS)"
