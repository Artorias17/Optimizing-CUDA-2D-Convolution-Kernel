#include <cudnn.h>

#include <cstdlib>
#include <iostream>

#include "conv.h"
#include "utils.h"

void launch_cudnn_conv(
    const float* d_input,
    const float* d_filter,
    float* d_output,
    ConvParams params
) {
    const int filter_width =
        2 * params.filter_radius + 1;

    cudnnHandle_t handle = nullptr;

    cudnnTensorDescriptor_t input_descriptor = nullptr;
    cudnnTensorDescriptor_t output_descriptor = nullptr;
    cudnnFilterDescriptor_t filter_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    CHECK_CUDNN(cudnnCreate(&handle));

    CHECK_CUDNN(
        cudnnCreateTensorDescriptor(&input_descriptor)
    );

    CHECK_CUDNN(
        cudnnCreateTensorDescriptor(&output_descriptor)
    );

    CHECK_CUDNN(
        cudnnCreateFilterDescriptor(&filter_descriptor)
    );

    CHECK_CUDNN(
        cudnnCreateConvolutionDescriptor(
            &convolution_descriptor
        )
    );

    // Input tensor: N = 1, C = 1, H, W.
    CHECK_CUDNN(
        cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            1,
            params.H,
            params.W
        )
    );

    // Filter tensor: K = 1, C = 1, R, S.
    CHECK_CUDNN(
        cudnnSetFilter4dDescriptor(
            filter_descriptor,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            1,
            1,
            filter_width,
            filter_width
        )
    );

    // Padding keeps the output dimensions equal to the input.
    CHECK_CUDNN(
        cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            params.filter_radius,
            params.filter_radius,
            1,
            1,
            1,
            1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        )
    );

    int output_n = 0;
    int output_c = 0;
    int output_h = 0;
    int output_w = 0;

    CHECK_CUDNN(
        cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor,
            input_descriptor,
            filter_descriptor,
            &output_n,
            &output_c,
            &output_h,
            &output_w
        )
    );

    if (
        output_n != 1 ||
        output_c != 1 ||
        output_h != params.H ||
        output_w != params.W
    ) {
        std::cerr
            << "Unexpected cuDNN output dimensions: "
            << output_n << " x "
            << output_c << " x "
            << output_h << " x "
            << output_w
            << std::endl;

        std::exit(EXIT_FAILURE);
    }

    CHECK_CUDNN(
        cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            output_n,
            output_c,
            output_h,
            output_w
        )
    );

    // Ask cuDNN for its recommended forward algorithm.
    cudnnConvolutionFwdAlgoPerf_t algorithm_result{};
    int returned_algorithm_count = 0;

    CHECK_CUDNN(
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            input_descriptor,
            filter_descriptor,
            convolution_descriptor,
            output_descriptor,
            1,
            &returned_algorithm_count,
            &algorithm_result
        )
    );

    if (
        returned_algorithm_count < 1 ||
        algorithm_result.status != CUDNN_STATUS_SUCCESS
    ) {
        std::cerr
            << "cuDNN could not select a convolution algorithm."
            << std::endl;

        std::exit(EXIT_FAILURE);
    }

    const cudnnConvolutionFwdAlgo_t algorithm =
        algorithm_result.algo;

    size_t workspace_size = 0;

    CHECK_CUDNN(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            input_descriptor,
            filter_descriptor,
            convolution_descriptor,
            output_descriptor,
            algorithm,
            &workspace_size
        )
    );

    void* d_workspace = nullptr;

    if (workspace_size > 0) {
        CHECK_CUDA(
            cudaMalloc(
                &d_workspace,
                workspace_size
            )
        );
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN(
        cudnnConvolutionForward(
            handle,
            &alpha,
            input_descriptor,
            d_input,
            filter_descriptor,
            d_filter,
            convolution_descriptor,
            algorithm,
            d_workspace,
            workspace_size,
            &beta,
            output_descriptor,
            d_output
        )
    );

    if (d_workspace != nullptr) {
        CHECK_CUDA(cudaFree(d_workspace));
    }

    CHECK_CUDNN(
        cudnnDestroyConvolutionDescriptor(
            convolution_descriptor
        )
    );

    CHECK_CUDNN(
        cudnnDestroyFilterDescriptor(
            filter_descriptor
        )
    );

    CHECK_CUDNN(
        cudnnDestroyTensorDescriptor(
            output_descriptor
        )
    );

    CHECK_CUDNN(
        cudnnDestroyTensorDescriptor(
            input_descriptor
        )
    );

    CHECK_CUDNN(cudnnDestroy(handle));
}