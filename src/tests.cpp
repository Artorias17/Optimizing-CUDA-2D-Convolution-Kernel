#include "tests.h"

#include <cuda_runtime.h>
#include <vector>

#include "conv.h"
#include "gpu_buffers.h"
#include "utils.h"

bool run_cpu_reference_test(const std::string &kernel,
                            const std::string &image_path,
                            const std::string &filter_path) {
    int H = 0, W = 0, filter_radius = 0;

    const std::vector<float> h_input  = load_image(image_path, H, W);
    const std::vector<float> h_filter = load_filter(filter_path, filter_radius);
    const int image_elements          = H * W;

    std::vector<float> h_cpu_output(image_elements, 0.0f);
    std::vector<float> h_gpu_output(image_elements, 0.0f);

    const ConvParams params{H, W, filter_radius, ZERO_PADDING};
    cpu_reference_conv(h_input.data(), h_filter.data(), h_cpu_output.data(), params);

    GpuBuffers gpu = copyToDevice(h_input, h_filter, image_elements);

    if (kernel == "cudnn")
        launch_cudnn_conv(gpu.d_input, gpu.d_filter, gpu.d_output, params);
    else
        launch_naive_conv(gpu.d_input, gpu.d_filter, gpu.d_output, params);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu_output.data(), gpu.d_output,
                          image_elements * sizeof(float), cudaMemcpyDeviceToHost));

    const bool passed = compare_outputs(
        h_cpu_output.data(), h_gpu_output.data(), image_elements, 1e-4f);

    freeDevice(gpu);
    return passed;
}
