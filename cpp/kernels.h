#pragma once

#include <cstdint>

namespace minillm {

void add_f32(const float* left, const float* right, float* output, std::int64_t count);
void mul_f32(const float* left, const float* right, float* output, std::int64_t count);
void scalar_mul_f32(const float* input, float scalar, float* output, std::int64_t count);
void matmul_f32(const float* left, const float* right, float* output,
                std::int64_t rows, std::int64_t inner, std::int64_t columns);
void batched_matmul_f32(const float* left, const float* right, float* output,
                        std::int64_t batches, std::int64_t rows,
                        std::int64_t inner, std::int64_t columns);
void softmax_rows_f32(const float* input, float* output,
                      std::int64_t rows, std::int64_t columns);
void relu_f32(const float* input, float* output, std::int64_t count);
void relu_backward_f32(const float* input, const float* grad_output,
                       float* grad_input, std::int64_t count);
void embedding_gather_f32(const float* table, const std::int32_t* indices,
                          float* output, std::int64_t vocab, std::int64_t width,
                          std::int64_t count);
void embedding_scatter_add_f32(const std::int32_t* indices, const float* grad_output,
                               float* grad_table, std::int64_t vocab,
                               std::int64_t width, std::int64_t count);
float cross_entropy_f32(const float* logits, const std::int32_t* targets,
                        std::int64_t rows, std::int64_t classes);
void cross_entropy_backward_f32(const float* logits, const std::int32_t* targets,
                                float* grad_logits, std::int64_t rows,
                                std::int64_t classes);
void adamw_f32(float* parameter, const float* gradient, float* first_moment,
               float* second_moment, std::int64_t count, float learning_rate,
               float beta1, float beta2, float epsilon, float weight_decay,
               std::int64_t step);

}  // namespace minillm

