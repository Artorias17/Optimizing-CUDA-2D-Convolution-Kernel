NVCC ?= nvcc

TARGET := conv
BUILD_DIR := build

CUDA_ARCH ?= sm_89

SOURCES := \
	src/main.cu \
	src/naive_conv.cu \
	src/cpu_reference.cu

OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))

NVCCFLAGS := \
	-std=c++17 \
	-O2 \
	-arch=$(CUDA_ARCH) \
	-Iinclude \
	-lineinfo \
	-Xcompiler=-Wall,-Wextra

LDFLAGS := -arch=$(CUDA_ARCH)

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(LDFLAGS) $(OBJECTS) -o $(TARGET)

$(BUILD_DIR)/%.o: src/%.cu
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