#ifndef CONV_H
#define CONV_H

// Maximum filter width reserved for CUDA constant memory.
#define MAX_CONST_MEM_FILTER 15

// Default CUDA thread block size.
#define BLOCK_SIZE 16

struct CudnnConvContext;

enum PadMode { ZERO_PADDING = 0, CLAMP_TO_EDGE = 1 };

struct ConvParams {
    int H;
    int W;
    int filter_radius;
    int pad_mode;
};

struct BenchmarkResult {
    float time_ms;
    float stddev_ms;
    float gflops;
    float bandwidth_gbs;
};

void launch_naive_conv(const float *d_input, const float *d_filter,
                       float *d_output, ConvParams params);

void launch_tiled_conv(const float *d_input, const float *d_filter,
                       float *d_output, ConvParams params);

void launch_const_mem_conv(const float *d_input, const float *d_filter,
                           float *d_output, ConvParams params);

void launch_cudnn_conv(const float *d_input, const float *d_filter,
                       float *d_output, ConvParams params);

CudnnConvContext *create_cudnn_conv_context(const float *d_input,
                                            const float *d_filter,
                                            float *d_output, ConvParams params);

void run_cudnn_conv(CudnnConvContext *context);

void destroy_cudnn_conv_context(CudnnConvContext *context);

void cpu_reference_conv(const float *input, const float *filter, float *output,
                        ConvParams params);

BenchmarkResult benchmark_naive_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs);

BenchmarkResult benchmark_tiled_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs);

BenchmarkResult benchmark_const_mem_conv(const float *d_input,
                                         const float *d_filter, float *d_output,
                                         ConvParams params, int warmup_runs,
                                         int timed_runs);

BenchmarkResult benchmark_cudnn_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs);

#endif