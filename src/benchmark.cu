#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "conv.h"
#include "utils.h"

BenchmarkResult benchmark_naive_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params,
    int warmup_runs,
    int timed_runs
) {
    if (warmup_runs < 0 || timed_runs <= 0) {
        std::cerr << "Invalid benchmark run counts."
                  << std::endl;

        std::exit(EXIT_FAILURE);
    }

    // Warmup executions are not included in the measurements.
    // They reduce first-launch and cache initialization effects.
    for (int run = 0; run < warmup_runs; ++run) {
        launch_naive_conv(
            d_input,
            d_filter,
            d_output,
            params
        );
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> execution_times;
    execution_times.reserve(timed_runs);

    CudaTimer timer;

    for (int run = 0; run < timed_runs; ++run) {
        timer.start();

        launch_naive_conv(
            d_input,
            d_filter,
            d_output,
            params
        );

        const float elapsed_ms = timer.stop();

        execution_times.push_back(elapsed_ms);
    }

    const float total_time = std::accumulate(
        execution_times.begin(),
        execution_times.end(),
        0.0f
    );

    const float mean_ms =
        total_time / static_cast<float>(timed_runs);

    float squared_difference_sum = 0.0f;

    for (const float time_ms : execution_times) {
        const float difference = time_ms - mean_ms;

        squared_difference_sum += difference * difference;
    }

    const float stddev_ms = std::sqrt(
        squared_difference_sum /
        static_cast<float>(timed_runs)
    );

    const int filter_width =
        2 * params.filter_radius + 1;

    const double output_pixels =
        static_cast<double>(params.H) *
        static_cast<double>(params.W);

    // One multiplication and one addition for every filter
    // element and output pixel.
    const double total_flops =
        2.0 *
        output_pixels *
        filter_width *
        filter_width;

    const double elapsed_seconds =
        static_cast<double>(mean_ms) / 1000.0;

    const float gflops = static_cast<float>(
        total_flops / elapsed_seconds / 1.0e9
    );

    // Estimated/effective traffic for the naive implementation:
    // K^2 input reads + K^2 filter reads + one output write.
    //
    // This is not measured DRAM traffic because GPU caches may
    // satisfy some of these loads.
    const double estimated_bytes =
        output_pixels *
        (2.0 * filter_width * filter_width + 1.0) *
        sizeof(float);

    const float bandwidth_gbs = static_cast<float>(
        estimated_bytes / elapsed_seconds / 1.0e9
    );

    return BenchmarkResult{
        mean_ms,
        stddev_ms,
        gflops,
        bandwidth_gbs
    };
}