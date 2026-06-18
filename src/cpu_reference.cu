#include <algorithm>

#include "conv.h"

void cpu_reference_conv(
    const float* input,
    const float* filter,
    float* output,
    ConvParams params
) {
    const int filter_width =
        2 * params.filter_radius + 1;

    for (int output_row = 0; output_row < params.H; ++output_row) {
        for (int output_col = 0; output_col < params.W; ++output_col) {
            float sum = 0.0f;

            for (
                int filter_row = -params.filter_radius;
                filter_row <= params.filter_radius;
                ++filter_row
            ) {
                for (
                    int filter_col = -params.filter_radius;
                    filter_col <= params.filter_radius;
                    ++filter_col
                ) {
                    int input_row = output_row + filter_row;
                    int input_col = output_col + filter_col;

                    float input_value = 0.0f;

                    if (params.pad_mode == CLAMP_TO_EDGE) {
                        input_row = std::max(
                            0,
                            std::min(input_row, params.H - 1)
                        );

                        input_col = std::max(
                            0,
                            std::min(input_col, params.W - 1)
                        );

                        input_value =
                            input[input_row * params.W + input_col];
                    } else {
                        const bool inside_image =
                            input_row >= 0 &&
                            input_row < params.H &&
                            input_col >= 0 &&
                            input_col < params.W;

                        if (inside_image) {
                            input_value =
                                input[input_row * params.W + input_col];
                        }
                    }

                    const int filter_index =
                        (filter_row + params.filter_radius)
                            * filter_width
                        + (filter_col + params.filter_radius);

                    sum += input_value * filter[filter_index];
                }
            }

            output[output_row * params.W + output_col] = sum;
        }
    }
}