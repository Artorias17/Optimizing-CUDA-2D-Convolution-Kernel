#include "gpu_buffers.h"
#include "utils.h"

GpuBuffers copyToDevice(
    const std::vector<float> &h_input,
    const std::vector<float> &h_filter,
    int image_elements
) {
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

void freeDevice(GpuBuffers &b) {
    CHECK_CUDA(cudaFree(b.d_input));
    CHECK_CUDA(cudaFree(b.d_filter));
    CHECK_CUDA(cudaFree(b.d_output));
    b = {};
}
