#include <cudnn.h>

#include <cstdlib>
#include <iostream>

#include "conv.h"
#include "utils.h"

static const char *cudnn_fwd_algo_name(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        return "IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        return "IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        return "GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        return "DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
        return "FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        return "FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        return "WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
        return "WINOGRAD_NONFUSED";
    default:
        return "UNKNOWN";
    }
}

#define CHECK_CUDNN(call)                                                      \
    do {                                                                       \
        cudnnStatus_t status = (call);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << "\nMessage: " << cudnnGetErrorString(status)          \
                      << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct CudnnConvContext {
    cudnnHandle_t handle = nullptr;

    cudnnTensorDescriptor_t input_descriptor = nullptr;
    cudnnTensorDescriptor_t output_descriptor = nullptr;
    cudnnFilterDescriptor_t filter_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t algorithm =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    void *d_workspace = nullptr;
    size_t workspace_size = 0;

    const float *d_input = nullptr;
    const float *d_filter = nullptr;
    float *d_output = nullptr;
};

CudnnConvContext *create_cudnn_conv_context(const float *d_input,
                                            const float *d_filter,
                                            float *d_output,
                                            ConvParams params) {
    if (params.pad_mode != ZERO_PADDING) {
        std::cerr << "cuDNN implementation currently supports "
                  << "ZERO_PADDING only." << std::endl;

        std::exit(EXIT_FAILURE);
    }

    auto *context = new CudnnConvContext{};

    context->d_input = d_input;
    context->d_filter = d_filter;
    context->d_output = d_output;

    const int filter_width = 2 * params.filter_radius + 1;

    CHECK_CUDNN(cudnnCreate(&context->handle));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&context->input_descriptor));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&context->output_descriptor));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&context->filter_descriptor));

    CHECK_CUDNN(
        cudnnCreateConvolutionDescriptor(&context->convolution_descriptor));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(context->input_descriptor,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, params.H, params.W));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(context->filter_descriptor,
                                           CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           1, 1, filter_width, filter_width));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        context->convolution_descriptor, params.filter_radius,
        params.filter_radius, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    int output_n = 0;
    int output_c = 0;
    int output_h = 0;
    int output_w = 0;

    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        context->convolution_descriptor, context->input_descriptor,
        context->filter_descriptor, &output_n, &output_c, &output_h,
        &output_w));

    if (output_n != 1 || output_c != 1 || output_h != params.H ||
        output_w != params.W) {
        std::cerr << "Unexpected cuDNN output dimensions: " << output_n << " x "
                  << output_c << " x " << output_h << " x " << output_w
                  << std::endl;

        std::exit(EXIT_FAILURE);
    }

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        context->output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        output_n, output_c, output_h, output_w));

    cudnnConvolutionFwdAlgoPerf_t algorithm_result{};
    int returned_algorithm_count = 0;

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        context->handle, context->input_descriptor, context->filter_descriptor,
        context->convolution_descriptor, context->output_descriptor, 1,
        &returned_algorithm_count, &algorithm_result));

    if (returned_algorithm_count < 1 ||
        algorithm_result.status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN could not select a forward algorithm." << std::endl;

        std::exit(EXIT_FAILURE);
    }

    context->algorithm = algorithm_result.algo;

    std::cout << "  cuDNN selected algorithm: "
              << cudnn_fwd_algo_name(context->algorithm) << std::endl;

    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        context->handle, context->input_descriptor, context->filter_descriptor,
        context->convolution_descriptor, context->output_descriptor,
        context->algorithm, &context->workspace_size));

    if (context->workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&context->d_workspace, context->workspace_size));
    }

    return context;
}

void run_cudnn_conv(CudnnConvContext *context) {
    if (context == nullptr) {
        std::cerr << "Cannot run cuDNN convolution with a null context."
                  << std::endl;

        std::exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(
        context->handle, &alpha, context->input_descriptor, context->d_input,
        context->filter_descriptor, context->d_filter,
        context->convolution_descriptor, context->algorithm,
        context->d_workspace, context->workspace_size, &beta,
        context->output_descriptor, context->d_output));
}

void destroy_cudnn_conv_context(CudnnConvContext *context) {
    if (context == nullptr) {
        return;
    }

    if (context->d_workspace != nullptr) {
        CHECK_CUDA(cudaFree(context->d_workspace));
    }

    if (context->convolution_descriptor != nullptr) {
        CHECK_CUDNN(
            cudnnDestroyConvolutionDescriptor(context->convolution_descriptor));
    }

    if (context->filter_descriptor != nullptr) {
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(context->filter_descriptor));
    }

    if (context->output_descriptor != nullptr) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(context->output_descriptor));
    }

    if (context->input_descriptor != nullptr) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(context->input_descriptor));
    }

    if (context->handle != nullptr) {
        CHECK_CUDNN(cudnnDestroy(context->handle));
    }

    delete context;
}

void launch_cudnn_conv(const float *d_input, const float *d_filter,
                       float *d_output, ConvParams params) {
    CudnnConvContext *context =
        create_cudnn_conv_context(d_input, d_filter, d_output, params);

    run_cudnn_conv(context);

    // Ensure the operation has finished before releasing
    // descriptors and workspace in this one-shot wrapper.
    CHECK_CUDA(cudaDeviceSynchronize());

    destroy_cudnn_conv_context(context);
}