#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << "\nError code: " << static_cast<int>(error)       \
                      << "\nMessage: " << cudaGetErrorString(error)        \
                      << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

class CudaTimer {
public:
    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event_));
        CHECK_CUDA(cudaEventCreate(&stop_event_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(start_event_, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_event_, stream));
        CHECK_CUDA(cudaEventSynchronize(stop_event_));

        float elapsed_ms = 0.0f;

        CHECK_CUDA(
            cudaEventElapsedTime(
                &elapsed_ms,
                start_event_,
                stop_event_
            )
        );

        return elapsed_ms;
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

inline void generate_random_image(
    float* image,
    int H,
    int W,
    unsigned int seed = 42
) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const int size = H * W;

    for (int i = 0; i < size; ++i) {
        image[i] = distribution(generator);
    }
}

inline void generate_random_filter(
    float* filter,
    int filter_radius,
    unsigned int seed = 123
) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const int filter_width = 2 * filter_radius + 1;
    const int filter_size = filter_width * filter_width;

    for (int i = 0; i < filter_size; ++i) {
        filter[i] = distribution(generator);
    }
}

inline bool compare_outputs(
    const float* reference,
    const float* output,
    int size,
    float tolerance = 1e-5f
) {
    float max_absolute_error = 0.0f;
    int max_error_index = -1;

    for (int i = 0; i < size; ++i) {
        const float error = std::fabs(reference[i] - output[i]);

        if (error > max_absolute_error) {
            max_absolute_error = error;
            max_error_index = i;
        }
    }

    std::cout << "Maximum absolute error: "
              << max_absolute_error
              << std::endl;

    if (max_absolute_error > tolerance) {
        std::cerr << "Output comparison failed at index "
                  << max_error_index
                  << std::endl;

        std::cerr << "Reference value: "
                  << reference[max_error_index]
                  << std::endl;

        std::cerr << "Output value: "
                  << output[max_error_index]
                  << std::endl;

        return false;
    }

    std::cout << "Output comparison passed."
              << std::endl;

    return true;
}

#endif