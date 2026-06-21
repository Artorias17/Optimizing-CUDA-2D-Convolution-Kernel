#include "gpu_buffers.h"
#include "utils.h"

GpuBuffers copyToDevice(const std::vector<float> &h_input,
                        const std::vector<float> &h_filter) {
    GpuBuffers buffers;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buffers.d_input),
                          h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buffers.d_filter),
                          h_filter.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buffers.d_output),
                          h_input.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(buffers.d_input, h_input.data(),
                          h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(buffers.d_filter, h_filter.data(),
                          h_filter.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    return buffers;
}

void freeDevice(GpuBuffers &b) {
    CHECK_CUDA(cudaFree(b.d_input));
    CHECK_CUDA(cudaFree(b.d_filter));
    CHECK_CUDA(cudaFree(b.d_output));
    b = {};
}
