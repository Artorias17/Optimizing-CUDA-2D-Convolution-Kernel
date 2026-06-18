#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "conv.h"
#include "utils.h"

__global__ void naive_conv_kernel(
    const float* input,
    const float* filter,
    float* output,
    int H,
    int W,
    int filter_radius,
    int pad_mode
) {
    const int output_row =
        blockIdx.y * blockDim.y + threadIdx.y;

    const int output_col =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Some threads can fall outside the image when H or W
    // is not divisible by the block dimensions.
    if (output_row >= H || output_col >= W) {
        return;
    }

    const int filter_width = 2 * filter_radius + 1;

    float sum = 0.0f;

    for (
        int filter_row = -filter_radius;
        filter_row <= filter_radius;
        ++filter_row
    ) {
        for (
            int filter_col = -filter_radius;
            filter_col <= filter_radius;
            ++filter_col
        ) {
            int input_row = output_row + filter_row;
            int input_col = output_col + filter_col;

            float input_value = 0.0f;

            if (pad_mode == CLAMP_TO_EDGE) {
                // Clamp an out-of-bounds coordinate to the nearest edge.
                input_row = max(0, min(input_row, H - 1));
                input_col = max(0, min(input_col, W - 1));

                input_value = input[input_row * W + input_col];
            } else {
                // ZERO_PADDING:
                // out-of-bounds pixels have value zero.
                const bool inside_image =
                    input_row >= 0 &&
                    input_row < H &&
                    input_col >= 0 &&
                    input_col < W;

                if (inside_image) {
                    input_value =
                        input[input_row * W + input_col];
                }
            }

            const int filter_index =
                (filter_row + filter_radius) * filter_width +
                (filter_col + filter_radius);

            sum += input_value * filter[filter_index];
        }
    }

    output[output_row * W + output_col] = sum;
}

void launch_naive_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params
) {
    if (
        params.H <= 0 ||
        params.W <= 0 ||
        params.filter_radius < 0
    ) {
        std::cerr << "Invalid convolution parameters."
                  << std::endl;

        std::exit(EXIT_FAILURE);
    }

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const dim3 grid(
        (params.W + block.x - 1) / block.x,
        (params.H + block.y - 1) / block.y
    );

    naive_conv_kernel<<<grid, block>>>(
        d_input,
        d_filter,
        d_output,
        params.H,
        params.W,
        params.filter_radius,
        params.pad_mode
    );

    // Checks whether the kernel launch itself failed.
    // Synchronization is handled by the caller or benchmark.
    CHECK_CUDA(cudaGetLastError());
}