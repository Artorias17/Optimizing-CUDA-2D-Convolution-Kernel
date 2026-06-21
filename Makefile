NVCC      ?= nvcc
TARGET    := conv
BUILD_DIR := build
CUDA_ARCH ?= sm_89

CUDNN_INCLUDE ?= $(wildcard .venv/lib/python3.11/site-packages/nvidia/cudnn/include)
CUDNN_LIB     ?= $(wildcard .venv/lib/python3.11/site-packages/nvidia/cudnn/lib)

CU_SOURCES  := $(wildcard src/*.cu)
CPP_SOURCES := $(wildcard src/*.cpp)
OBJECTS     := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES)) \
               $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
HEADERS     := $(wildcard include/*.h)

NVCCFLAGS := \
	-std=c++17 \
	-O2 \
	-arch=$(CUDA_ARCH) \
	-Iinclude \
	$(if $(CUDNN_INCLUDE),-I$(CUDNN_INCLUDE)) \
	-lineinfo \
	-Xcompiler=-Wall,-Wextra

LDFLAGS := \
	-arch=$(CUDA_ARCH) \
	$(if $(CUDNN_LIB),-L$(CUDNN_LIB) -Xlinker=-rpath=$(CUDNN_LIB))

LDLIBS := -l:libcudnn.so.9

.PHONY: all debug clean

all: $(TARGET)

debug: NVCCFLAGS := \
	-std=c++17 \
	-O0 -g -G \
	-arch=$(CUDA_ARCH) \
	-Iinclude \
	$(if $(CUDNN_INCLUDE),-I$(CUDNN_INCLUDE)) \
	-Xcompiler=-Wall,-Wextra

debug: clean all

$(TARGET): $(OBJECTS)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
