#include "kernels.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
