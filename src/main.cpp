#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "conv.h"
#include "gpu_buffers.h"
#include "tests.h"
#include "utils.h"

static const std::string KERNELS[] = {"naive", "cudnn"};

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

static void run_all_benchmarks(const std::string &image_path,
                               const std::string &filter_path,
                               int runs) {
    constexpr int warmup_runs = 3;

    int H = 0, W = 0, filter_radius = 0;

    const std::vector<float> h_input  = load_image(image_path, H, W);
    const std::vector<float> h_filter = load_filter(filter_path, filter_radius);
    const int image_elements          = H * W;
    const int filter_width            = 2 * filter_radius + 1;

    GpuBuffers gpu = copyToDevice(h_input, h_filter, image_elements);

    const ConvParams params{H, W, filter_radius, ZERO_PADDING};

    for (const std::string &kernel : KERNELS) {
        BenchmarkResult result{};

        if (kernel == "naive") {
            result = benchmark_naive_conv(gpu.d_input, gpu.d_filter, gpu.d_output,
                                          params, warmup_runs, runs);
        } else {
            result = benchmark_cudnn_conv(gpu.d_input, gpu.d_filter, gpu.d_output,
                                          params, warmup_runs, runs);
        }

        print_benchmark_result(kernel, image_path, filter_path,
                               H, W, filter_width, runs, result);
    }

    freeDevice(gpu);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

static void print_usage(const char *program_name) {
    std::cout << "Usage:\n"
              << "  " << program_name
              << " --image PATH --filter PATH [--runs 50]\n\n"
              << "Options:\n"
              << "  --image PATH   Grayscale PGM image file\n"
              << "  --filter PATH  Plain-text filter file\n"
              << "  --runs VALUE   Number of timed samples per kernel (default 50)\n"
              << "  --help, -h     Show this help\n";
}

int main(int argc, char **argv) {
    try {
        std::string image_path, filter_path;
        int runs = 50;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto val = [&] {
                if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
                return std::string(argv[i]);
            };

            if      (arg == "--image")               image_path  = val();
            else if (arg == "--filter")              filter_path = val();
            else if (arg == "--runs")                runs = std::stoi(val());
            else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return EXIT_SUCCESS; }
            else throw std::runtime_error("Unknown argument: " + arg);
        }

        if (image_path.empty() || filter_path.empty())
            throw std::runtime_error("--image and --filter are required.");
        if (runs <= 0)
            throw std::runtime_error("Number of runs must be positive.");

        print_device_information();

        for (const std::string &kernel : KERNELS) {
            std::cout << "Verifying " << kernel << " against CPU reference...\n";
            if (!run_cpu_reference_test(kernel, image_path, filter_path)) {
                std::cerr << kernel << " correctness check FAILED — aborting.\n";
                return EXIT_FAILURE;
            }
            std::cout << "Passed.\n\n";
        }

        run_all_benchmarks(image_path, filter_path, runs);

        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}
