NVCC ?= nvcc

TARGET := conv
BUILD_DIR := build

CUDA_ARCH ?= sm_89
LDLIBS := -lcudnn

SOURCES := \
	src/main.cu \
	src/naive_conv.cu \
	src/cpu_reference.cu \
	src/benchmark.cu \
	src/cudnn_conv.cu

OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))

HEADERS := $(wildcard include/*.h)

CUDNN_INCLUDE ?= $(wildcard .venv/lib/python3.11/site-packages/nvidia/cudnn/include)
CUDNN_LIB     ?= $(wildcard .venv/lib/python3.11/site-packages/nvidia/cudnn/lib)

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
	$(if $(CUDNN_LIB),-L$(CUDNN_LIB))

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(LDFLAGS) $(OBJECTS) -o $(TARGET) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

debug: NVCCFLAGS := \
	-std=c++17 \
	-O0 \
	-g \
	-G \
	-arch=$(CUDA_ARCH) \
	-Iinclude \
	-Xcompiler=-Wall,-Wextra

debug: clean all

clean:
	rm -rf $(BUILD_DIR) $(TARGET)