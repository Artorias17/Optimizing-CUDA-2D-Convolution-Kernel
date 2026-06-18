#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
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

int main() {
    print_device_information();

    std::cout << "Running 5 x 5 identity filter test...\n\n";

    const bool test_passed = run_identity_filter_test();

    if (!test_passed) {
        std::cerr << "\nNaive convolution test failed."
                  << std::endl;

        return EXIT_FAILURE;
    }

    std::cout << "\nNaive convolution test passed."
              << std::endl;

    return EXIT_SUCCESS;
}