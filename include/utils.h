#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conv.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << "\nError code: " << static_cast<int>(error)           \
                      << "\nMessage: " << cudaGetErrorString(error)            \
                      << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

class CudaTimer {
  public:
    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event_));
        CHECK_CUDA(cudaEventCreate(&stop_event_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(start_event_, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_event_, stream));
        CHECK_CUDA(cudaEventSynchronize(stop_event_));

        float elapsed_ms = 0.0f;

        CHECK_CUDA(
            cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));

        return elapsed_ms;
    }

    CudaTimer(const CudaTimer &) = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

  private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

// Loads an ASCII PGM (P2) image. Pixel values are normalised to [0, 1].
// Sets H and W from the file header.
inline std::vector<float> load_image(const std::string &path, int &H, int &W) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open image: " + path);

    std::string magic;
    f >> magic;
    if (magic != "P2")
        throw std::runtime_error("Only ASCII PGM (P2) supported: " + path);

    // Skip comment lines and return the next whitespace-separated token.
    auto next_token = [&]() -> std::string {
        std::string token;
        while (f >> token) {
            if (token[0] == '#') {
                std::string rest;
                std::getline(f, rest);
                continue;
            }
            return token;
        }
        throw std::runtime_error("Unexpected end of file: " + path);
    };

    W = std::stoi(next_token());
    H = std::stoi(next_token());
    const int maxval = std::stoi(next_token());

    const int n = H * W;
    std::vector<float> pixels(n);
    for (int i = 0; i < n; ++i)
        pixels[i] = std::stof(next_token()) / static_cast<float>(maxval);

    return pixels;
}

// Loads a filter from a plain-text file.
// Format: first line is "rows cols", followed by filter values row by row.
// Filter must be square with odd dimensions. Sets filter_radius accordingly.
inline std::vector<float> load_filter(const std::string &path,
                                      int &filter_radius) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open filter: " + path);

    int rows = 0;
    int cols = 0;
    if (!(f >> rows >> cols))
        throw std::runtime_error("Invalid filter header: " + path);

    if (rows != cols)
        throw std::runtime_error("Filter must be square: " + path);
    if (rows % 2 == 0)
        throw std::runtime_error("Filter dimensions must be odd: " + path);

    filter_radius = (rows - 1) / 2;

    const int n = rows * cols;
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i)
        if (!(f >> values[i]))
            throw std::runtime_error("Unexpected end of filter data: " + path);

    return values;
}

inline bool compare_outputs(const float *reference, const float *output,
                            int size, float tolerance = 1e-5f) {
    float max_absolute_error = 0.0f;
    int max_error_index = -1;

    for (int i = 0; i < size; ++i) {
        const float error = std::fabs(reference[i] - output[i]);

        if (error > max_absolute_error) {
            max_absolute_error = error;
            max_error_index = i;
        }
    }

    std::cout << "Maximum absolute error: " << max_absolute_error << std::endl;

    if (max_absolute_error > tolerance) {
        std::cerr << "Output comparison failed at index " << max_error_index
                  << std::endl;

        std::cerr << "Reference value: " << reference[max_error_index]
                  << std::endl;

        std::cerr << "Output value: " << output[max_error_index] << std::endl;

        return false;
    }

    std::cout << "Output comparison passed." << std::endl;

    return true;
}

inline void print_device_information() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU was detected." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "CUDA devices detected: " << device_count << "\n\n";

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));

    std::cout << "Device 0: " << properties.name << '\n'
              << "Compute capability: " << properties.major << '.'
              << properties.minor << '\n'
              << "Global memory: "
              << properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n\n";
}

inline void print_benchmark_result(const std::string &kernel,
                                   const std::string &image_path,
                                   const std::string &filter_path, int H, int W,
                                   int filter_width, int timed_runs,
                                   const BenchmarkResult &result) {
    const int filter_radius = filter_width / 2;
    std::cout << '\n'
              << kernel << " benchmark\n"
              << "Image:  " << image_path << " (" << H << " x " << W << ")\n"
              << "Filter: " << filter_path << " (" << filter_width << " x "
              << filter_width << ")\n"
              << "Timed runs:                    " << timed_runs << '\n'
              << "Mean execution time:           " << result.time_ms << " ms\n"
              << "Standard deviation:            " << result.stddev_ms
              << " ms\n"
              << "Performance:                   " << result.gflops
              << " GFLOPS\n"
              << "Estimated effective bandwidth: " << result.bandwidth_gbs
              << " GB/s\n";
    std::printf(
        "RESULT kernel=%s H=%d W=%d filter_radius=%d filter_width=%d"
        " runs=%d mean_ms=%.6f stddev_ms=%.6f gflops=%.4f bw_gbs=%.4f\n",
        kernel.c_str(), H, W, filter_radius, filter_width, timed_runs,
        result.time_ms, result.stddev_ms, result.gflops, result.bandwidth_gbs);
}

#endif