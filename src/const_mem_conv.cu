#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#include "conv.h"
#include "utils.h"

// Covers up to MAX_CONST_MEM_FILTER × MAX_CONST_MEM_FILTER filter values.
__constant__ float d_const_filter[MAX_CONST_MEM_FILTER * MAX_CONST_MEM_FILTER];

// Output tile side length — same as tiled_conv so both kernels are
// directly comparable.
#define TILE_WIDTH 16

// Tiled shared-memory load + constant-memory filter read.
__global__ void const_mem_conv_kernel(const float *d_input, float *d_output,
                                      int H, int W, int filter_radius,
                                      int pad_mode) {
    extern __shared__ float s_tile[];

    const int tile_dim = TILE_WIDTH + 2 * filter_radius;

    const int in_row = blockIdx.y * TILE_WIDTH - filter_radius + threadIdx.y;
    const int in_col = blockIdx.x * TILE_WIDTH - filter_radius + threadIdx.x;

    float val = 0.0f;
    if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
        val = d_input[in_row * W + in_col];
    } else if (pad_mode == CLAMP_TO_EDGE) {
        const int cr = max(0, min(in_row, H - 1));
        const int cc = max(0, min(in_col, W - 1));
        val = d_input[cr * W + cc];
    }

    s_tile[threadIdx.y * tile_dim + threadIdx.x] = val;
    __syncthreads();

    if (threadIdx.y >= TILE_WIDTH || threadIdx.x >= TILE_WIDTH)
        return;

    const int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (out_row >= H || out_col >= W)
        return;

    const int filter_width = 2 * filter_radius + 1;
    float sum = 0.0f;

    for (int fr = 0; fr < filter_width; ++fr) {
        for (int fc = 0; fc < filter_width; ++fc) {
            sum += s_tile[(threadIdx.y + fr) * tile_dim + (threadIdx.x + fc)]
                   * d_const_filter[fr * filter_width + fc];
        }
    }

    d_output[out_row * W + out_col] = sum;
}

void launch_const_mem_conv(const float *d_input, const float *d_filter,
                           float *d_output, ConvParams params) {
    if (params.H <= 0 || params.W <= 0 || params.filter_radius < 0) {
        std::cerr << "Invalid convolution parameters." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int filter_width = 2 * params.filter_radius + 1;
    CHECK_CUDA(cudaMemcpyToSymbol(d_const_filter, d_filter,
                                  filter_width * filter_width * sizeof(float),
                                  0, cudaMemcpyDeviceToDevice));

    const int tile_dim = TILE_WIDTH + 2 * params.filter_radius;
    const dim3 block(tile_dim, tile_dim);
    const dim3 grid((params.W + TILE_WIDTH - 1) / TILE_WIDTH,
                    (params.H + TILE_WIDTH - 1) / TILE_WIDTH);
    const size_t shared_mem = static_cast<size_t>(tile_dim) * tile_dim * sizeof(float);

    const_mem_conv_kernel<<<grid, block, shared_mem>>>(
        d_input, d_output, params.H, params.W, params.filter_radius,
        params.pad_mode);

    CHECK_CUDA(cudaGetLastError());
}
