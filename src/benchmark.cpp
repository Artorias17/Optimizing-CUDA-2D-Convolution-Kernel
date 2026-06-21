#include <cmath>
#include <numeric>
#include <vector>

#include "conv.h"
#include "utils.h"

static BenchmarkResult compute_result(const std::vector<float> &times,
                                      ConvParams params) {
    const int n = static_cast<int>(times.size());
    const float mean_ms = std::accumulate(times.begin(), times.end(), 0.0f) / n;

    float sq_diff = 0.0f;
    for (float t : times)
        sq_diff += (t - mean_ms) * (t - mean_ms);
    const float stddev_ms = std::sqrt(sq_diff / n);

    const int filter_width = 2 * params.filter_radius + 1;
    const double pixels = static_cast<double>(params.H) * params.W;
    const double seconds = mean_ms / 1000.0;
    const float gflops = static_cast<float>(2.0 * pixels * filter_width *
                                            filter_width / seconds / 1e9);
    const float bandwidth_gbs =
        static_cast<float>(pixels * (2.0 * filter_width * filter_width + 1.0) *
                           sizeof(float) / seconds / 1e9);

    return BenchmarkResult{mean_ms, stddev_ms, gflops, bandwidth_gbs};
}

BenchmarkResult benchmark_naive_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs) {
    for (int i = 0; i < warmup_runs; ++i) {
        launch_naive_conv(d_input, d_filter, d_output, params);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> times;
    times.reserve(timed_runs);

    for (int run = 0; run < timed_runs; ++run) {
        timer.start();
        launch_naive_conv(d_input, d_filter, d_output, params);
        times.push_back(timer.stop());
    }

    return compute_result(times, params);
}

BenchmarkResult benchmark_tiled_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs) {
    for (int i = 0; i < warmup_runs; ++i)
        launch_tiled_conv(d_input, d_filter, d_output, params);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> times;
    times.reserve(timed_runs);

    for (int run = 0; run < timed_runs; ++run) {
        timer.start();
        launch_tiled_conv(d_input, d_filter, d_output, params);
        times.push_back(timer.stop());
    }

    return compute_result(times, params);
}

BenchmarkResult benchmark_const_mem_conv(const float *d_input,
                                         const float *d_filter, float *d_output,
                                         ConvParams params, int warmup_runs,
                                         int timed_runs) {
    for (int i = 0; i < warmup_runs; ++i)
        launch_const_mem_conv(d_input, d_filter, d_output, params);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> times;
    times.reserve(timed_runs);

    for (int run = 0; run < timed_runs; ++run) {
        timer.start();
        launch_const_mem_conv(d_input, d_filter, d_output, params);
        times.push_back(timer.stop());
    }

    return compute_result(times, params);
}

BenchmarkResult benchmark_cudnn_conv(const float *d_input,
                                     const float *d_filter, float *d_output,
                                     ConvParams params, int warmup_runs,
                                     int timed_runs) {
    CudnnConvContext *ctx =
        create_cudnn_conv_context(d_input, d_filter, d_output, params);

    for (int i = 0; i < warmup_runs; ++i)
        run_cudnn_conv(ctx);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> times;
    times.reserve(timed_runs);

    for (int run = 0; run < timed_runs; ++run) {
        timer.start();
        run_cudnn_conv(ctx);
        times.push_back(timer.stop());
    }

    destroy_cudnn_conv_context(ctx);
    return compute_result(times, params);
}
