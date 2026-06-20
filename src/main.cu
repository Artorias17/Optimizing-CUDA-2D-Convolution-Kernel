#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conv.h"
#include "utils.h"

// ---------------------------------------------------------------------------
// GPU buffer management
// ---------------------------------------------------------------------------

struct GpuBuffers {
    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;
};

static GpuBuffers alloc_and_upload(const std::vector<float> &h_input,
                                   const std::vector<float> &h_filter,
                                   int image_elements) {
    GpuBuffers b;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&b.d_input),
                          h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&b.d_filter),
                          h_filter.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&b.d_output),
                          static_cast<size_t>(image_elements) * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(b.d_input, h_input.data(),
                          h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b.d_filter, h_filter.data(),
                          h_filter.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    return b;
}

static void free_gpu_buffers(GpuBuffers &b) {
    CHECK_CUDA(cudaFree(b.d_input));
    CHECK_CUDA(cudaFree(b.d_filter));
    CHECK_CUDA(cudaFree(b.d_output));
    b = {};
}

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------

static void print_device_information() {
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

// ---------------------------------------------------------------------------
// Correctness tests
// ---------------------------------------------------------------------------

static void print_matrix(const std::vector<float> &matrix, int H, int W) {
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col)
            std::cout << matrix[row * W + col] << '\t';
        std::cout << '\n';
    }
}

static bool run_identity_filter_test() {
    constexpr int H = 5;
    constexpr int W = 5;
    constexpr int filter_radius = 1;
    constexpr int filter_width = 2 * filter_radius + 1;
    constexpr int image_elements = H * W;
    constexpr int filter_elements = filter_width * filter_width;

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements, 0.0f);
    std::vector<float> h_output(image_elements, 0.0f);

    for (int i = 0; i < image_elements; ++i)
        h_input[i] = static_cast<float>(i + 1);

    // Identity filter: centre element = 1, rest = 0.
    h_filter[filter_radius * filter_width + filter_radius] = 1.0f;

    const ConvParams params{H, W, filter_radius, ZERO_PADDING};
    GpuBuffers gpu = alloc_and_upload(h_input, h_filter, image_elements);

    launch_naive_conv(gpu.d_input, gpu.d_filter, gpu.d_output, params);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), gpu.d_output,
                          image_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "Naive convolution output:\n";
    print_matrix(h_output, H, W);
    std::cout << '\n';

    const bool passed =
        compare_outputs(h_input.data(), h_output.data(), image_elements, 1e-5f);

    free_gpu_buffers(gpu);
    return passed;
}

// Runs `kernel` on a small 7x9 image and compares against the CPU reference.
static bool run_cpu_reference_test(const std::string &kernel) {
    constexpr int H = 7;
    constexpr int W = 9;
    constexpr int filter_radius = 2;
    constexpr int filter_width = 2 * filter_radius + 1;
    constexpr int image_elements = H * W;
    constexpr int filter_elements = filter_width * filter_width;

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements);
    std::vector<float> h_cpu_output(image_elements, 0.0f);
    std::vector<float> h_gpu_output(image_elements, 0.0f);

    generate_random_image(h_input.data(), H, W, 42);
    generate_random_filter(h_filter.data(), filter_radius, 123);

    const ConvParams params{H, W, filter_radius, ZERO_PADDING};
    cpu_reference_conv(h_input.data(), h_filter.data(), h_cpu_output.data(),
                       params);

    GpuBuffers gpu = alloc_and_upload(h_input, h_filter, image_elements);

    if (kernel == "cudnn")
        launch_cudnn_conv(gpu.d_input, gpu.d_filter, gpu.d_output, params);
    else
        launch_naive_conv(gpu.d_input, gpu.d_filter, gpu.d_output, params);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu_output.data(), gpu.d_output,
                          image_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const bool passed = compare_outputs(
        h_cpu_output.data(), h_gpu_output.data(), image_elements, 1e-4f);

    free_gpu_buffers(gpu);
    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

struct CliOptions {
    bool run_tests = false;
    bool run_benchmark = false;
    bool show_help = false;

    std::string kernel = "naive";
    std::string output_path;

    int H = 1024;
    int W = 1024;
    int filter_radius = 2;
    int runs = 50;
};

static void print_benchmark_result(const std::string &kernel, int H, int W,
                                   int filter_width, int timed_runs,
                                   const BenchmarkResult &result) {
    std::cout << '\n'
              << kernel << " benchmark\n"
              << "Image:  " << H << " x " << W << '\n'
              << "Filter: " << filter_width << " x " << filter_width << '\n'
              << "Timed runs:                    " << timed_runs << '\n'
              << "Mean execution time:           " << result.time_ms << " ms\n"
              << "Standard deviation:            " << result.stddev_ms
              << " ms\n"
              << "Performance:                   " << result.gflops
              << " GFLOPS\n"
              << "Estimated effective bandwidth: " << result.bandwidth_gbs
              << " GB/s\n";
}

static BenchmarkResult run_benchmark(const CliOptions &opts) {
    constexpr int warmup_runs = 3;

    const int filter_width = 2 * opts.filter_radius + 1;
    const int image_elements = opts.H * opts.W;
    const int filter_elements = filter_width * filter_width;

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements);

    generate_random_image(h_input.data(), opts.H, opts.W, 42);
    generate_random_filter(h_filter.data(), opts.filter_radius, 123);

    GpuBuffers gpu = alloc_and_upload(h_input, h_filter, image_elements);

    const ConvParams params{opts.H, opts.W, opts.filter_radius, ZERO_PADDING};

    BenchmarkResult result{};

    if (opts.kernel == "naive") {
        result = benchmark_naive_conv(gpu.d_input, gpu.d_filter, gpu.d_output,
                                      params, warmup_runs, opts.runs);
    } else {
        CudnnConvContext *ctx = create_cudnn_conv_context(
            gpu.d_input, gpu.d_filter, gpu.d_output, params);
        result = benchmark_cudnn_conv(ctx, params, warmup_runs, opts.runs);
        destroy_cudnn_conv_context(ctx);
    }

    print_benchmark_result(opts.kernel, opts.H, opts.W, filter_width, opts.runs,
                           result);

    free_gpu_buffers(gpu);
    return result;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void print_usage(const char *program_name) {
    std::cout << "Usage:\n"
              << "  " << program_name << " --test\n"
              << "  " << program_name << " --kernel [naive|cudnn]"
              << " --H 1024 --W 1024 --filter-radius 2 --runs 50\n\n"
              << "Options:\n"
              << "  --test                 Run correctness tests\n"
              << "  --kernel VALUE         naive or cudnn\n"
              << "  --H VALUE              Image height\n"
              << "  --W VALUE              Image width\n"
              << "  --filter-radius VALUE  Filter radius\n"
              << "  --runs VALUE           Number of timed samples\n"
              << "  --output PATH          Append benchmark result to CSV\n"
              << "  --help, -h             Show this help\n";
}

static CliOptions parse_cli_arguments(int argc, char **argv) {
    CliOptions options;

    if (argc == 1) {
        options.run_tests = true;
        options.run_benchmark = true;
        return options;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        auto read_value = [&](const std::string &option_name) {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + option_name);
            return std::string(argv[++i]);
        };

        if (argument == "--test") {
            options.run_tests = true;
        } else if (argument == "--kernel") {
            options.kernel = read_value(argument);
            options.run_benchmark = true;
        } else if (argument == "--H") {
            options.H = std::stoi(read_value(argument));
        } else if (argument == "--W") {
            options.W = std::stoi(read_value(argument));
        } else if (argument == "--filter-radius") {
            options.filter_radius = std::stoi(read_value(argument));
        } else if (argument == "--runs") {
            options.runs = std::stoi(read_value(argument));
        } else if (argument == "--output") {
            options.output_path = read_value(argument);
        } else if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.run_benchmark && options.kernel != "naive" &&
        options.kernel != "cudnn")
        throw std::runtime_error("Kernel must be 'naive' or 'cudnn'.");

    if (options.H <= 0 || options.W <= 0)
        throw std::runtime_error("Image dimensions must be positive.");

    if (options.filter_radius < 0 || options.filter_radius > MAX_FILTER_RADIUS)
        throw std::runtime_error("Filter radius must be between 0 and " +
                                 std::to_string(MAX_FILTER_RADIUS) + ".");

    if (options.runs <= 0)
        throw std::runtime_error("Number of runs must be positive.");

    if (!options.output_path.empty() && !options.run_benchmark)
        throw std::runtime_error("--output can only be used with a benchmark.");

    return options;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

static void append_benchmark_result_to_csv(const std::string &output_path,
                                           const std::string &kernel_name,
                                           int H, int W, int filter_radius,
                                           const BenchmarkResult &result) {
    const std::filesystem::path file_path(output_path);

    if (file_path.has_parent_path())
        std::filesystem::create_directories(file_path.parent_path());

    const bool write_header = !std::filesystem::exists(file_path) ||
                              std::filesystem::file_size(file_path) == 0;

    std::ofstream output_file(output_path, std::ios::app);

    if (!output_file.is_open())
        throw std::runtime_error("Could not open output file: " + output_path);

    if (write_header)
        output_file << "kernel,H,W,filter_radius,"
                    << "mean_ms,stddev_ms,gflops,bandwidth_gbs\n";

    output_file << std::fixed << std::setprecision(6) << kernel_name << ',' << H
                << ',' << W << ',' << filter_radius << ',' << result.time_ms
                << ',' << result.stddev_ms << ',' << result.gflops << ','
                << result.bandwidth_gbs << '\n';

    std::cout << "Result appended to: " << output_path << std::endl;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    try {
        const CliOptions options = parse_cli_arguments(argc, argv);

        if (options.show_help) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        print_device_information();

        if (options.run_tests) {
            std::cout << "Running 5x5 identity filter test...\n\n";
            if (!run_identity_filter_test()) {
                std::cerr << "Identity filter test failed." << std::endl;
                return EXIT_FAILURE;
            }
            std::cout << "Identity filter test passed.\n";

            for (const std::string &kernel : {"naive", "cudnn"}) {
                std::cout << "\nRunning 7x9 CPU reference test (" << kernel
                          << ")...\n\n";
                if (!run_cpu_reference_test(kernel)) {
                    std::cerr << kernel << " CPU reference test failed."
                              << std::endl;
                    return EXIT_FAILURE;
                }
                std::cout << kernel << " CPU reference test passed.\n";
            }
        }

        if (options.run_benchmark) {
            const BenchmarkResult result = run_benchmark(options);

            if (!options.output_path.empty()) {
                append_benchmark_result_to_csv(
                    options.output_path, options.kernel, options.H, options.W,
                    options.filter_radius, result);
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}
