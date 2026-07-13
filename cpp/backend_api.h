#pragma once

#include <cstdint>

#if defined(_WIN32)
#define MIMILLM_EXPORT __declspec(dllexport)
#else
#define MIMILLM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

MIMILLM_EXPORT const char* mimillm_last_error();
MIMILLM_EXPORT const char* mimillm_compiler_info();
MIMILLM_EXPORT int mimillm_set_num_threads(std::int32_t threads);
MIMILLM_EXPORT std::int32_t mimillm_get_num_threads();
MIMILLM_EXPORT int mimillm_add_f32(const float*, const float*, float*, std::int64_t);
MIMILLM_EXPORT int mimillm_mul_f32(const float*, const float*, float*, std::int64_t);
MIMILLM_EXPORT int mimillm_scalar_mul_f32(const float*, float, float*, std::int64_t);
MIMILLM_EXPORT int mimillm_permute_f32(const float*, float*, const std::int64_t*, const std::int64_t*, std::int64_t);
MIMILLM_EXPORT int mimillm_broadcast_binary_f32(
    const float*, const float*, float*, const std::int64_t*, std::int64_t,
    const std::int64_t*, std::int64_t, const std::int64_t*, std::int64_t,
    std::int32_t);
MIMILLM_EXPORT int mimillm_broadcast_binary_backward_f32(
    const float*, const float*, const float*, float*, float*,
    const std::int64_t*, std::int64_t, const std::int64_t*, std::int64_t,
    const std::int64_t*, std::int64_t, std::int32_t);
MIMILLM_EXPORT int mimillm_matmul_f32(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_batched_matmul_f32(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_softmax_rows_f32(const float*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_softmax_backward_f32(const float*, const float*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_sum_rows_f32(const float*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_sum_rows_backward_f32(const float*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_relu_f32(const float*, float*, std::int64_t);
MIMILLM_EXPORT int mimillm_relu_backward_f32(const float*, const float*, float*, std::int64_t);
MIMILLM_EXPORT int mimillm_embedding_gather_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_embedding_scatter_add_f32(const std::int32_t*, const float*, float*, std::int64_t, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_cross_entropy_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_cross_entropy_backward_f32(const float*, const std::int32_t*, float*, std::int64_t, std::int64_t);
MIMILLM_EXPORT int mimillm_adamw_f32(float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, std::int64_t);

}
