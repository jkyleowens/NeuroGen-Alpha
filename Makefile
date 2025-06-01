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
INCLUDE_DIRS := -I./src/cuda -I$(CUDA_HOME)/include
LIB_DIRS := -L$(CUDA_HOME)/lib64

# Compilation flags
CXXFLAGS := -std=c++17 -O2 $(INCLUDE_DIRS) --expt-relaxed-constexpr -DUSE_CUDA=1
LDFLAGS := $(LIB_DIRS) -lcurand

# Source files
CU_SRCS = \
    src/cuda/RandomStateInit.cu \
    src/cuda/KernelLaunchWrappers.cu \
    src/cuda/NeuronUpdateKernel.cu \
    src/cuda/NeuronSpikingKernels.cu \
    src/cuda/SynapseInputKernel.cu \
    src/cuda/STDPKernel.cu \
    src/cuda/NetworkCUDA.cu

CPP_SRCS = \
    main.cpp

# Object files
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)

# Target binary
TARGET = neural_sim

all: $(TARGET)

# Link all object files
$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -dc $< -o $@

# Compile C++ sources
%.o: %.cpp
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(TARGET)
