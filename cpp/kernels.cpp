#include "kernels.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mimillm {

void add_f32(const float* left, const float* right, float* output, std::int64_t count) {
    global_thread_pool().parallel_for(0, count, 32768, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) output[i] = left[i] + right[i];
    });
}

void mul_f32(const float* left, const float* right, float* output, std::int64_t count) {
    global_thread_pool().parallel_for(0, count, 32768, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) output[i] = left[i] * right[i];
    });
}

void scalar_mul_f32(const float* input, float scalar, float* output, std::int64_t count) {
    global_thread_pool().parallel_for(0, count, 32768, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) output[i] = input[i] * scalar;
    });
}

void permute_f32(const float* input, float* output, const std::int64_t* shape,
                 const std::int64_t* axes, std::int64_t dimensions) {
    std::int64_t count = 1;
    std::vector<std::int64_t> source_strides(dimensions, 1);
    std::vector<std::int64_t> output_shape(dimensions, 1);
    std::vector<std::int64_t> output_strides(dimensions, 1);
    for (std::int64_t axis = dimensions - 1; axis >= 0; --axis) {
        if (shape[axis] < 0 || axes[axis] < 0 || axes[axis] >= dimensions) {
            throw std::invalid_argument("invalid permute dimensions");
        }
        if (axis + 1 < dimensions) {
            source_strides[axis] = source_strides[axis + 1] * shape[axis + 1];
        }
        count *= shape[axis];
        output_shape[axis] = shape[axes[axis]];
    }
    for (std::int64_t axis = dimensions - 2; axis >= 0; --axis) {
        output_strides[axis] = output_strides[axis + 1] * output_shape[axis + 1];
    }
    global_thread_pool().parallel_for(0, count, 4096, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t flat = begin; flat < end; ++flat) {
            std::int64_t source_index = 0;
            for (std::int64_t output_axis = 0; output_axis < dimensions; ++output_axis) {
                const auto coordinate = (flat / output_strides[output_axis]) % output_shape[output_axis];
                source_index += coordinate * source_strides[axes[output_axis]];
            }
            output[flat] = input[source_index];
        }
    });
}

namespace {

struct BroadcastLayout {
    std::vector<std::int64_t> output_shape;
    std::vector<std::int64_t> output_strides;
    std::vector<std::int64_t> operand_shape;
    std::vector<std::int64_t> operand_strides;
    std::int64_t offset{0};
    std::int64_t output_count{1};
    std::int64_t operand_count{1};
};

BroadcastLayout broadcast_layout(
    const std::int64_t* operand_shape, std::int64_t operand_dimensions,
    const std::int64_t* output_shape, std::int64_t output_dimensions
) {
    if (operand_dimensions < 0 || output_dimensions < operand_dimensions) {
        throw std::invalid_argument("invalid broadcast dimensions");
    }
    BroadcastLayout layout;
    layout.output_shape.assign(output_shape, output_shape + output_dimensions);
    layout.operand_shape.assign(operand_shape, operand_shape + operand_dimensions);
    layout.output_strides.assign(output_dimensions, 1);
    layout.operand_strides.assign(operand_dimensions, 1);
    layout.offset = output_dimensions - operand_dimensions;
    for (std::int64_t axis = output_dimensions - 1; axis >= 0; --axis) {
        if (layout.output_shape[axis] <= 0) throw std::invalid_argument("invalid output shape");
        if (axis + 1 < output_dimensions) {
            layout.output_strides[axis] = layout.output_strides[axis + 1]
                * layout.output_shape[axis + 1];
        }
        layout.output_count *= layout.output_shape[axis];
    }
    for (std::int64_t axis = operand_dimensions - 1; axis >= 0; --axis) {
        const auto output_axis = layout.offset + axis;
        const auto dimension = layout.operand_shape[axis];
        if (dimension <= 0 || (dimension != 1 && dimension != layout.output_shape[output_axis])) {
            throw std::invalid_argument("incompatible broadcast shape");
        }
        if (axis + 1 < operand_dimensions) {
            layout.operand_strides[axis] = layout.operand_strides[axis + 1]
                * layout.operand_shape[axis + 1];
        }
        layout.operand_count *= dimension;
    }
    return layout;
}

std::int64_t broadcast_index(std::int64_t flat, const BroadcastLayout& layout) {
    std::int64_t index = 0;
    for (std::int64_t axis = 0; axis < static_cast<std::int64_t>(layout.operand_shape.size()); ++axis) {
        if (layout.operand_shape[axis] == 1) continue;
        const auto output_axis = layout.offset + axis;
        const auto coordinate = (flat / layout.output_strides[output_axis])
            % layout.output_shape[output_axis];
        index += coordinate * layout.operand_strides[axis];
    }
    return index;
}

float binary_value(float left, float right, std::int32_t operation) {
    switch (operation) {
        case 0: return left + right;
        case 1: return left - right;
        case 2: return left * right;
        case 3: return left / right;
        default: throw std::invalid_argument("unknown binary operation");
    }
}

}  // namespace

void broadcast_binary_f32(
    const float* left, const float* right, float* output,
    const std::int64_t* left_shape, std::int64_t left_dimensions,
    const std::int64_t* right_shape, std::int64_t right_dimensions,
    const std::int64_t* output_shape, std::int64_t output_dimensions,
    std::int32_t operation
) {
    const auto left_layout = broadcast_layout(
        left_shape, left_dimensions, output_shape, output_dimensions
    );
    const auto right_layout = broadcast_layout(
        right_shape, right_dimensions, output_shape, output_dimensions
    );
    global_thread_pool().parallel_for(
        0, left_layout.output_count, 4096,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t flat = begin; flat < end; ++flat) {
                output[flat] = binary_value(
                    left[broadcast_index(flat, left_layout)],
                    right[broadcast_index(flat, right_layout)],
                    operation
                );
            }
        }
    );
}

void broadcast_binary_backward_f32(
    const float* left, const float* right, const float* grad_output,
    float* grad_left, float* grad_right,
    const std::int64_t* left_shape, std::int64_t left_dimensions,
    const std::int64_t* right_shape, std::int64_t right_dimensions,
    const std::int64_t* output_shape, std::int64_t output_dimensions,
    std::int32_t operation
) {
    const auto left_layout = broadcast_layout(
        left_shape, left_dimensions, output_shape, output_dimensions
    );
    const auto right_layout = broadcast_layout(
        right_shape, right_dimensions, output_shape, output_dimensions
    );
    std::fill(grad_left, grad_left + left_layout.operand_count, 0.0F);
    std::fill(grad_right, grad_right + right_layout.operand_count, 0.0F);
    for (std::int64_t flat = 0; flat < left_layout.output_count; ++flat) {
        const auto left_index = broadcast_index(flat, left_layout);
        const auto right_index = broadcast_index(flat, right_layout);
        const float upstream = grad_output[flat];
        const float left_value = left[left_index];
        const float right_value = right[right_index];
        switch (operation) {
            case 0:
                grad_left[left_index] += upstream;
                grad_right[right_index] += upstream;
                break;
            case 1:
                grad_left[left_index] += upstream;
                grad_right[right_index] -= upstream;
                break;
            case 2:
                grad_left[left_index] += upstream * right_value;
                grad_right[right_index] += upstream * left_value;
                break;
            case 3:
                grad_left[left_index] += upstream / right_value;
                grad_right[right_index] -= upstream * left_value / (right_value * right_value);
                break;
            default:
                throw std::invalid_argument("unknown binary operation");
        }
    }
}

void matmul_f32(const float* left, const float* right, float* output,
                std::int64_t rows, std::int64_t inner, std::int64_t columns) {
    global_thread_pool().parallel_for(0, rows, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            std::fill(output + row * columns, output + (row + 1) * columns, 0.0F);
            for (std::int64_t k = 0; k < inner; ++k) {
                const float value = left[row * inner + k];
                for (std::int64_t column = 0; column < columns; ++column) {
                    output[row * columns + column] += value * right[k * columns + column];
                }
            }
        }
    });
}

void batched_matmul_f32(const float* left, const float* right, float* output,
                        std::int64_t batches, std::int64_t rows,
                        std::int64_t inner, std::int64_t columns) {
    const std::int64_t left_size = rows * inner;
    const std::int64_t right_size = inner * columns;
    const std::int64_t output_size = rows * columns;
    global_thread_pool().parallel_for(0, batches * rows, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t task = begin; task < end; ++task) {
            const std::int64_t batch = task / rows;
            const std::int64_t row = task % rows;
            float* destination = output + batch * output_size + row * columns;
            std::fill(destination, destination + columns, 0.0F);
            for (std::int64_t k = 0; k < inner; ++k) {
                const float value = left[batch * left_size + row * inner + k];
                for (std::int64_t column = 0; column < columns; ++column) {
                    destination[column] += value * right[batch * right_size + k * columns + column];
                }
            }
        }
    });
}

void softmax_rows_f32(const float* input, float* output,
                      std::int64_t rows, std::int64_t columns) {
    global_thread_pool().parallel_for(0, rows, 16, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            const float* source = input + row * columns;
            float* destination = output + row * columns;
            const float maximum = *std::max_element(source, source + columns);
            double denominator = 0.0;
            for (std::int64_t column = 0; column < columns; ++column) {
                destination[column] = std::exp(source[column] - maximum);
                denominator += destination[column];
            }
            for (std::int64_t column = 0; column < columns; ++column) {
                destination[column] = static_cast<float>(destination[column] / denominator);
            }
        }
    });
}

void softmax_backward_f32(const float* output, const float* grad_output,
                          float* grad_input, std::int64_t rows,
                          std::int64_t columns) {
    global_thread_pool().parallel_for(0, rows, 16, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            const auto offset = row * columns;
            double dot = 0.0;
            for (std::int64_t column = 0; column < columns; ++column) {
                dot += grad_output[offset + column] * output[offset + column];
            }
            for (std::int64_t column = 0; column < columns; ++column) {
                grad_input[offset + column] = output[offset + column]
                    * (grad_output[offset + column] - static_cast<float>(dot));
            }
        }
    });
}

void sum_rows_f32(const float* input, float* output,
                  std::int64_t rows, std::int64_t columns) {
    global_thread_pool().parallel_for(0, rows, 64, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            double total = 0.0;
            for (std::int64_t column = 0; column < columns; ++column) {
                total += input[row * columns + column];
            }
            output[row] = static_cast<float>(total);
        }
    });
}

void sum_rows_backward_f32(const float* grad_output, float* grad_input,
                           std::int64_t rows, std::int64_t columns) {
    global_thread_pool().parallel_for(0, rows, 64, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            std::fill(
                grad_input + row * columns,
                grad_input + (row + 1) * columns,
                grad_output[row]
            );
        }
    });
}

void relu_f32(const float* input, float* output, std::int64_t count) {
    global_thread_pool().parallel_for(0, count, 32768, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) output[i] = std::max(input[i], 0.0F);
    });
}

void relu_backward_f32(const float* input, const float* grad_output,
                       float* grad_input, std::int64_t count) {
    global_thread_pool().parallel_for(0, count, 32768, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            grad_input[i] = input[i] > 0.0F ? grad_output[i] : 0.0F;
        }
    });
}

void embedding_gather_f32(const float* table, const std::int32_t* indices,
                          float* output, std::int64_t vocab, std::int64_t width,
                          std::int64_t count) {
    for (std::int64_t row = 0; row < count; ++row) {
        const auto index = static_cast<std::int64_t>(indices[row]);
        if (index < 0 || index >= vocab) throw std::out_of_range("embedding index out of range");
        std::copy(table + index * width, table + (index + 1) * width, output + row * width);
    }
}

void embedding_scatter_add_f32(const std::int32_t* indices, const float* grad_output,
                               float* grad_table, std::int64_t vocab,
                               std::int64_t width, std::int64_t count) {
    std::fill(grad_table, grad_table + vocab * width, 0.0F);
    for (std::int64_t row = 0; row < count; ++row) {
        const auto index = static_cast<std::int64_t>(indices[row]);
        if (index < 0 || index >= vocab) throw std::out_of_range("embedding index out of range");
        for (std::int64_t column = 0; column < width; ++column) {
            grad_table[index * width + column] += grad_output[row * width + column];
        }
    }
}

float cross_entropy_f32(const float* logits, const std::int32_t* targets,
                        std::int64_t rows, std::int64_t classes) {
    double total = 0.0;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto target = static_cast<std::int64_t>(targets[row]);
        if (target < 0 || target >= classes) throw std::out_of_range("target out of range");
        const float* values = logits + row * classes;
        const float maximum = *std::max_element(values, values + classes);
        double denominator = 0.0;
        for (std::int64_t column = 0; column < classes; ++column) {
            denominator += std::exp(values[column] - maximum);
        }
        total += std::log(denominator) + maximum - values[target];
    }
    return static_cast<float>(total / rows);
}

void cross_entropy_backward_f32(const float* logits, const std::int32_t* targets,
                                float* grad_logits, std::int64_t rows,
                                std::int64_t classes) {
    softmax_rows_f32(logits, grad_logits, rows, classes);
    const float scale = 1.0F / static_cast<float>(rows);
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto target = static_cast<std::int64_t>(targets[row]);
        if (target < 0 || target >= classes) throw std::out_of_range("target out of range");
        grad_logits[row * classes + target] -= 1.0F;
    }
    for (std::int64_t i = 0; i < rows * classes; ++i) grad_logits[i] *= scale;
}

void adamw_f32(float* parameter, const float* gradient, float* first_moment,
               float* second_moment, std::int64_t count, float learning_rate,
               float beta1, float beta2, float epsilon, float weight_decay,
               std::int64_t step) {
    const double correction1 = 1.0 - std::pow(static_cast<double>(beta1), step);
    const double correction2 = 1.0 - std::pow(static_cast<double>(beta2), step);
    for (std::int64_t i = 0; i < count; ++i) {
        first_moment[i] = beta1 * first_moment[i] + (1.0F - beta1) * gradient[i];
        second_moment[i] = beta2 * second_moment[i] + (1.0F - beta2) * gradient[i] * gradient[i];
        const double first_hat = first_moment[i] / correction1;
        const double second_hat = second_moment[i] / correction2;
        const double update = first_hat / (std::sqrt(second_hat) + epsilon) + weight_decay * parameter[i];
        parameter[i] -= static_cast<float>(learning_rate * update);
    }
}

}  // namespace mimillm
