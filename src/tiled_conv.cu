#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "conv.h"
#include "utils.h"

#define TILE_WIDTH 16

__global__ void tiled_conv_kernel(const float *d_input, const float *d_filter,
                                  float *d_output, int H, int W,
                                  int filter_radius, int pad_mode) {
    extern __shared__ float s_tile[];

    // Each thread maps to one cell in the shared tile.
    // The tile's top-left corner in global input coordinates is offset by
    // -filter_radius so the halo pixels are included.
    int in_row = blockIdx.y * TILE_WIDTH - filter_radius + threadIdx.y;
    int in_col = blockIdx.x * TILE_WIDTH - filter_radius + threadIdx.x;

    // Load one input pixel into shared memory, applying boundary condition.
    // ZERO_PADDING: val stays 0.0f for out-of-bounds coordinates.
    float val = 0.0f;
    if (in_row < 0 || in_row >= H || in_col < 0 && in_col >= W) {
        // Clamp out-of-bounds coordinates to the nearest edge pixel.
        if (pad_mode == CLAMP_TO_EDGE) {
            in_row = max(0, min(in_row, H - 1));
            in_col = max(0, min(in_col, W - 1));
        }
    }

    val = d_input[in_row * W + in_col];
    s_tile[threadIdx.y * blockDim.x + threadIdx.x] = val;

    // Wait for all threads in the block to write its pixel to shared
    // the memory.
    __syncthreads();

    // Only the inner TILE_WIDTH × TILE_WIDTH threads compute output pixels.
    if (threadIdx.y >= TILE_WIDTH || threadIdx.x >= TILE_WIDTH)
        return;

    // Map the inner thread to its output pixel.
    const int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Guard against output pixels outside the image (edge blocks).
    if (out_row >= H || out_col >= W)
        return;

    // Read input values from shared memory.
    const int filter_width = 2 * filter_radius + 1;
    float sum = 0.0f;

    for (int fr = 0; fr < filter_width; ++fr) {
        for (int fc = 0; fc < filter_width; ++fc) {
            sum +=
                s_tile[(threadIdx.y + fr) * blockDim.x + (threadIdx.x + fc)] *
                d_filter[fr * filter_width + fc];
        }
    }

    d_output[out_row * W + out_col] = sum;
}

void launch_tiled_conv(const float *d_input, const float *d_filter,
                       float *d_output, ConvParams params) {
    if (params.H <= 0 || params.W <= 0 || params.filter_radius < 0) {
        std::cerr << "Invalid convolution parameters." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int tile_dim = TILE_WIDTH + 2 * params.filter_radius;

    // Block covers the full shared tile (output + halo); grid covers the image
    // in steps of TILE_WIDTH (output pixels only).
    const dim3 block(tile_dim, tile_dim);
    const dim3 grid((params.W + TILE_WIDTH - 1) / TILE_WIDTH,
                    (params.H + TILE_WIDTH - 1) / TILE_WIDTH);
    const size_t shared_mem =
        static_cast<size_t>(tile_dim) * tile_dim * sizeof(float);

    tiled_conv_kernel<<<grid, block, shared_mem>>>(
        d_input, d_filter, d_output, params.H, params.W, params.filter_radius,
        params.pad_mode);

    CHECK_CUDA(cudaGetLastError());
}
