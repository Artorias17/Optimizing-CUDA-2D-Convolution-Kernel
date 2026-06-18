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

void print_device_information() {
    int device_count = 0;

    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU was detected."
                  << std::endl;

        std::exit(EXIT_FAILURE);
    }

    std::cout << "CUDA devices detected: "
              << device_count
              << "\n\n";

    cudaDeviceProp properties{};

    CHECK_CUDA(
        cudaGetDeviceProperties(&properties, 0)
    );

    std::cout << "Device 0: "
              << properties.name
              << '\n';

    std::cout << "Compute capability: "
              << properties.major
              << "."
              << properties.minor
              << '\n';

    std::cout << "Global memory: "
              << properties.totalGlobalMem
                     / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n\n";
}

void print_matrix(
    const std::vector<float>& matrix,
    int H,
    int W
) {
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            std::cout << matrix[row * W + col]
                      << '\t';
        }

        std::cout << '\n';
    }
}

bool run_identity_filter_test() {
    constexpr int H = 5;
    constexpr int W = 5;
    constexpr int filter_radius = 1;

    const int filter_width =
    2 * filter_radius + 1;

const int image_elements = H * W;

const int filter_elements =
    filter_width * filter_width;

    const size_t image_bytes =
        image_elements * sizeof(float);

    const size_t filter_bytes =
        filter_elements * sizeof(float);

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements, 0.0f);
    std::vector<float> h_output(image_elements, 0.0f);

    // Create a simple 5 x 5 image containing values 1 to 25.
    for (int i = 0; i < image_elements; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    // Identity filter:
    //
    // 0 0 0
    // 0 1 0
    // 0 0 0
    //
    // The output should therefore be identical to the input.
    const int filter_center =
        filter_radius * filter_width + filter_radius;

    h_filter[filter_center] = 1.0f;

    float* d_input = nullptr;
    float* d_filter = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(
        cudaMalloc(
            reinterpret_cast<void**>(&d_input),
            image_bytes
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            reinterpret_cast<void**>(&d_filter),
            filter_bytes
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            reinterpret_cast<void**>(&d_output),
            image_bytes
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_input,
            h_input.data(),
            image_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_filter,
            h_filter.data(),
            filter_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const ConvParams params{
        H,
        W,
        filter_radius,
        ZERO_PADDING
    };

    launch_naive_conv(
        d_input,
        d_filter,
        d_output,
        params
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(
        cudaMemcpy(
            h_output.data(),
            d_output,
            image_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    std::cout << "Naive convolution output:\n";
    print_matrix(h_output, H, W);
    std::cout << '\n';

    const bool passed = compare_outputs(
        h_input.data(),
        h_output.data(),
        image_elements,
        1e-5f
    );

   CHECK_CUDA(cudaFree(d_input));
CHECK_CUDA(cudaFree(d_filter));
CHECK_CUDA(cudaFree(d_output));

return passed;
}

bool run_random_reference_test() {
    constexpr int H = 7;
    constexpr int W = 9;
    constexpr int filter_radius = 2;

    constexpr int filter_width =
        2 * filter_radius + 1;

    constexpr int image_elements = H * W;

    constexpr int filter_elements =
        filter_width * filter_width;

    const size_t image_bytes =
        image_elements * sizeof(float);

    const size_t filter_bytes =
        filter_elements * sizeof(float);

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements);
    std::vector<float> h_cpu_output(image_elements, 0.0f);
    std::vector<float> h_gpu_output(image_elements, 0.0f);

    generate_random_image(
        h_input.data(),
        H,
        W,
        42
    );

    generate_random_filter(
        h_filter.data(),
        filter_radius,
        123
    );

    const ConvParams params{
        H,
        W,
        filter_radius,
        ZERO_PADDING
    };

    cpu_reference_conv(
        h_input.data(),
        h_filter.data(),
        h_cpu_output.data(),
        params
    );

    float* d_input = nullptr;
    float* d_filter = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        image_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_filter),
        filter_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        image_bytes
    ));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        image_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_filter,
        h_filter.data(),
        filter_bytes,
        cudaMemcpyHostToDevice
    ));

    launch_naive_conv(
        d_input,
        d_filter,
        d_output,
        params
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_gpu_output.data(),
        d_output,
        image_bytes,
        cudaMemcpyDeviceToHost
    ));

    const bool passed = compare_outputs(
        h_cpu_output.data(),
        h_gpu_output.data(),
        image_elements,
        1e-4f
    );

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));

    return passed;
}

BenchmarkResult run_naive_benchmark(
    int H,
    int W,
    int filter_radius,
    int timed_runs
) {
    constexpr int warmup_runs = 3;

    const int filter_width =
        2 * filter_radius + 1;

    const int image_elements = H * W;

    const int filter_elements =
        filter_width * filter_width;

    const size_t image_bytes =
        static_cast<size_t>(image_elements) * sizeof(float);

    const size_t filter_bytes =
        static_cast<size_t>(filter_elements) * sizeof(float);

    std::vector<float> h_input(image_elements);
    std::vector<float> h_filter(filter_elements);

    generate_random_image(
        h_input.data(),
        H,
        W,
        42
    );

    generate_random_filter(
        h_filter.data(),
        filter_radius,
        123
    );

    float* d_input = nullptr;
    float* d_filter = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        image_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_filter),
        filter_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        image_bytes
    ));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        image_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_filter,
        h_filter.data(),
        filter_bytes,
        cudaMemcpyHostToDevice
    ));

    const ConvParams params{
        H,
        W,
        filter_radius,
        ZERO_PADDING
    };

    const BenchmarkResult result =
        benchmark_naive_conv(
            d_input,
            d_filter,
            d_output,
            params,
            warmup_runs,
            timed_runs
        );

    std::cout << "\nNaive CUDA benchmark\n";
    std::cout << "Image: "
              << H << " x " << W << '\n';

    std::cout << "Filter: "
              << filter_width
              << " x "
              << filter_width
              << '\n';

    std::cout << "Timed runs: "
              << timed_runs
              << '\n';

    std::cout << "Mean execution time: "
              << result.time_ms
              << " ms\n";

    std::cout << "Standard deviation: "
              << result.stddev_ms
              << " ms\n";

    std::cout << "Performance: "
              << result.gflops
              << " GFLOPS\n";

    std::cout << "Estimated effective bandwidth: "
              << result.bandwidth_gbs
              << " GB/s\n";

    CHECK_CUDA(cudaFree(d_input));
CHECK_CUDA(cudaFree(d_filter));
CHECK_CUDA(cudaFree(d_output));

return result;
}

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

void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name << " --test\n"
        << "  " << program_name
        << " --kernel naive"
        << " --H 1024"
        << " --W 1024"
        << " --filter-radius 2"
        << " --runs 50\n\n"
        << "Options:\n"
        << "  --test                 Run correctness tests\n"
        << "  --kernel naive         Select CUDA kernel\n"
        << "  --H VALUE              Image height\n"
        << "  --W VALUE              Image width\n"
        << "  --filter-radius VALUE  Filter radius\n"
        << "  --runs VALUE           Number of timed samples\n"
        << "  --output PATH          Append benchmark result to CSV\n"
        << "  --help, -h             Show this help\n";
}

CliOptions parse_cli_arguments(int argc, char** argv) {
    CliOptions options;

    // Preserve the old behavior when no arguments are supplied.
    if (argc == 1) {
        options.run_tests = true;
        options.run_benchmark = true;

        return options;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        auto read_value = [&](const std::string& option_name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "Missing value for " + option_name
                );
            }

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
            options.filter_radius =
                std::stoi(read_value(argument));
        } else if (argument == "--runs") {
    options.runs = std::stoi(read_value(argument));
} else if (argument == "--output") {
    options.output_path = read_value(argument);
} else if (
    argument == "--help" ||
    argument == "-h"
) {
    options.show_help = true;
} else {
    throw std::runtime_error(
        "Unknown argument: " + argument
    );
}
    }

    if (options.run_benchmark && options.kernel != "naive") {
        throw std::runtime_error(
            "Only the naive kernel is currently available."
        );
    }

    if (options.H <= 0 || options.W <= 0) {
        throw std::runtime_error(
            "Image dimensions must be positive."
        );
    }

    if (
        options.filter_radius < 0 ||
        options.filter_radius > MAX_FILTER_RADIUS
    ) {
        throw std::runtime_error(
            "Filter radius must be between 0 and "
            + std::to_string(MAX_FILTER_RADIUS)
            + "."
        );
    }

    if (options.runs <= 0) {
        throw std::runtime_error(
            "Number of runs must be positive."
        );
    }
    if (
    !options.output_path.empty() &&
    !options.run_benchmark
) {
    throw std::runtime_error(
        "--output can only be used with a benchmark."
    );
}

    return options;
}

void append_benchmark_result_to_csv(
    const std::string& output_path,
    const std::string& kernel_name,
    int H,
    int W,
    int filter_radius,
    const BenchmarkResult& result
) {
    const std::filesystem::path file_path(output_path);

    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(
            file_path.parent_path()
        );
    }

    const bool write_header =
        !std::filesystem::exists(file_path) ||
        std::filesystem::file_size(file_path) == 0;

    std::ofstream output_file(
        output_path,
        std::ios::app
    );

    if (!output_file.is_open()) {
        throw std::runtime_error(
            "Could not open output file: " + output_path
        );
    }

    if (write_header) {
        output_file
            << "kernel,H,W,filter_radius,"
            << "mean_ms,stddev_ms,gflops,bandwidth_gbs\n";
    }

    output_file
        << std::fixed
        << std::setprecision(6)
        << kernel_name << ','
        << H << ','
        << W << ','
        << filter_radius << ','
        << result.time_ms << ','
        << result.stddev_ms << ','
        << result.gflops << ','
        << result.bandwidth_gbs
        << '\n';

    std::cout
        << "Result appended to: "
        << output_path
        << std::endl;
}

int main(int argc, char** argv) {
    try {
        const CliOptions options =
            parse_cli_arguments(argc, argv);

        if (options.show_help) {
            print_usage(argv[0]);

            return EXIT_SUCCESS;
        }

        print_device_information();

        if (options.run_tests) {
            std::cout
                << "Running 5 x 5 identity filter test...\n\n";

            const bool identity_test_passed =
                run_identity_filter_test();

            if (!identity_test_passed) {
                std::cerr
                    << "\nNaive convolution identity test failed."
                    << std::endl;

                return EXIT_FAILURE;
            }

            std::cout
                << "\nNaive convolution identity test passed."
                << std::endl;

            std::cout
                << "\nRunning random 7 x 9 image "
                << "with 5 x 5 filter...\n\n";

            const bool random_test_passed =
                run_random_reference_test();

            if (!random_test_passed) {
                std::cerr
                    << "\nRandom CPU/GPU comparison failed."
                    << std::endl;

                return EXIT_FAILURE;
            }

            std::cout
                << "\nRandom CPU/GPU comparison passed."
                << std::endl;
        }

        if (options.run_benchmark) {
    const BenchmarkResult result =
        run_naive_benchmark(
            options.H,
            options.W,
            options.filter_radius,
            options.runs
        );

    if (!options.output_path.empty()) {
        append_benchmark_result_to_csv(
            options.output_path,
            options.kernel,
            options.H,
            options.W,
            options.filter_radius,
            result
        );
    }
}

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << "\n\n";

        print_usage(argv[0]);

        return EXIT_FAILURE;
    }
}