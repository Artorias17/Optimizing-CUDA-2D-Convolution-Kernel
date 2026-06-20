#ifndef GPU_BUFFERS_H
#define GPU_BUFFERS_H

#include <vector>

struct GpuBuffers {
    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;
};

GpuBuffers copyToDevice(const std::vector<float> &h_input,
                        const std::vector<float> &h_filter, int image_elements);

void freeDevice(GpuBuffers &b);

#endif
