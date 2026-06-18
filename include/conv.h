#ifndef CONV_H
#define CONV_H

// Largest filter used in the benchmark: 11 x 11.
#define MAX_FILTER_SIZE 11

// Radius corresponding to an 11 x 11 filter:
// radius = (filter_size - 1) / 2 = 5.
#define MAX_FILTER_RADIUS 5

// Maximum filter width reserved for CUDA constant memory.
#define MAX_CONST_MEM_FILTER 15

// Default CUDA thread block size.
#define BLOCK_SIZE 16

enum PadMode {
    ZERO_PADDING = 0,
    CLAMP_TO_EDGE = 1
};

struct ConvParams {
    int H;
    int W;
    int filter_radius;
    int pad_mode;
};

struct BenchmarkResult {
    float time_ms;
    float gflops;
    float bandwidth_gbs;
};

// Person A
void launch_naive_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params
);

// Person B
void launch_tiled_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params
);

// Person B
void launch_const_mem_conv(
    const float* d_input,
    const float* h_filter,
    float* d_output,
    ConvParams params
);

// Person A
void launch_cudnn_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params
);

#endif