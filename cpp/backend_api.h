#pragma once

#include <cstdint>

#if defined(_WIN32)
#define MINILLM_EXPORT __declspec(dllexport)
#else
#define MINILLM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

MINILLM_EXPORT const char* minillm_last_error();
MINILLM_EXPORT const char* minillm_compiler_info();
MINILLM_EXPORT int minillm_set_num_threads(std::int32_t threads);
MINILLM_EXPORT std::int32_t minillm_get_num_threads();
MINILLM_EXPORT int minillm_add_f32(const float*, const float*, float*, std::int64_t);
MINILLM_EXPORT int minillm_mul_f32(const float*, const float*, float*, std::int64_t);
MINILLM_EXPORT int minillm_scalar_mul_f32(const float*, float, float*, std::int64_t);
MINILLM_EXPORT int minillm_matmul_f32(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_batched_matmul_f32(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_softmax_rows_f32(const float*, float*, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_relu_f32(const float*, float*, std::int64_t);
MINILLM_EXPORT int minillm_relu_backward_f32(const float*, const float*, float*, std::int64_t);
MINILLM_EXPORT int minillm_embedding_gather_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_embedding_scatter_add_f32(const std::int32_t*, const float*, float*, std::int64_t, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_cross_entropy_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_cross_entropy_backward_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t);
MINILLM_EXPORT int minillm_adamw_f32(float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, std::int64_t);

}

