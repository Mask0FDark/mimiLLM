#include "backend_api.h"
#include "kernels.h"
#include "thread_pool.h"

#include <algorithm>
#include <exception>
#include <string>
#include <thread>

namespace {
thread_local std::string last_error;

bool pointers(std::initializer_list<const void*> values) {
    return std::all_of(values.begin(), values.end(), [](const void* value) { return value != nullptr; });
}

template <typename Function>
int guarded(Function&& function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return 1;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
}  // namespace

extern "C" {

const char* mimillm_last_error() { return last_error.c_str(); }

const char* mimillm_compiler_info() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown compiler";
#endif
}

int mimillm_set_num_threads(std::int32_t threads) {
    return guarded([&] { mimillm::global_thread_pool().set_active_threads(threads); });
}

std::int32_t mimillm_get_num_threads() { return mimillm::global_thread_pool().active_threads(); }

int mimillm_add_f32(const float* a, const float* b, float* out, std::int64_t n) {
    return guarded([&] { require(pointers({a, b, out}), "null pointer"); require(n >= 0, "count must be non-negative"); mimillm::add_f32(a, b, out, n); });
}
int mimillm_mul_f32(const float* a, const float* b, float* out, std::int64_t n) {
    return guarded([&] { require(pointers({a, b, out}), "null pointer"); require(n >= 0, "count must be non-negative"); mimillm::mul_f32(a, b, out, n); });
}
int mimillm_scalar_mul_f32(const float* a, float s, float* out, std::int64_t n) {
    return guarded([&] { require(pointers({a, out}), "null pointer"); require(n >= 0, "count must be non-negative"); mimillm::scalar_mul_f32(a, s, out, n); });
}
int mimillm_permute_f32(const float* a, float* out, const std::int64_t* shape, const std::int64_t* axes, std::int64_t dimensions) {
    return guarded([&] { require(pointers({a, out, shape, axes}), "null pointer"); require(dimensions > 0, "dimensions must be positive"); mimillm::permute_f32(a, out, shape, axes, dimensions); });
}
int mimillm_broadcast_binary_f32(
    const float* left, const float* right, float* out,
    const std::int64_t* left_shape, std::int64_t left_dimensions,
    const std::int64_t* right_shape, std::int64_t right_dimensions,
    const std::int64_t* output_shape, std::int64_t output_dimensions,
    std::int32_t operation
) {
    return guarded([&] {
        require(pointers({left, right, out, output_shape}), "null pointer");
        require(output_dimensions > 0, "output dimensions must be positive");
        mimillm::broadcast_binary_f32(
            left, right, out, left_shape, left_dimensions, right_shape,
            right_dimensions, output_shape, output_dimensions, operation
        );
    });
}
int mimillm_broadcast_binary_backward_f32(
    const float* left, const float* right, const float* grad_output,
    float* grad_left, float* grad_right,
    const std::int64_t* left_shape, std::int64_t left_dimensions,
    const std::int64_t* right_shape, std::int64_t right_dimensions,
    const std::int64_t* output_shape, std::int64_t output_dimensions,
    std::int32_t operation
) {
    return guarded([&] {
        require(pointers({left, right, grad_output, grad_left, grad_right, output_shape}), "null pointer");
        require(output_dimensions > 0, "output dimensions must be positive");
        mimillm::broadcast_binary_backward_f32(
            left, right, grad_output, grad_left, grad_right,
            left_shape, left_dimensions, right_shape, right_dimensions,
            output_shape, output_dimensions, operation
        );
    });
}
int mimillm_matmul_f32(const float* a, const float* b, float* out, std::int64_t r, std::int64_t k, std::int64_t c) {
    return guarded([&] { require(pointers({a, b, out}), "null pointer"); require(r >= 0 && k > 0 && c >= 0, "invalid matmul dimensions"); mimillm::matmul_f32(a, b, out, r, k, c); });
}
int mimillm_batched_matmul_f32(const float* a, const float* b, float* out, std::int64_t batches, std::int64_t r, std::int64_t k, std::int64_t c) {
    return guarded([&] { require(pointers({a, b, out}), "null pointer"); require(batches > 0 && r >= 0 && k > 0 && c >= 0, "invalid batched matmul dimensions"); mimillm::batched_matmul_f32(a, b, out, batches, r, k, c); });
}
int mimillm_softmax_rows_f32(const float* a, float* out, std::int64_t r, std::int64_t c) {
    return guarded([&] { require(pointers({a, out}), "null pointer"); require(r >= 0 && c > 0, "invalid softmax dimensions"); mimillm::softmax_rows_f32(a, out, r, c); });
}
int mimillm_softmax_backward_f32(const float* out, const float* grad, float* grad_input, std::int64_t r, std::int64_t c) {
    return guarded([&] { require(pointers({out, grad, grad_input}), "null pointer"); require(r >= 0 && c > 0, "invalid softmax dimensions"); mimillm::softmax_backward_f32(out, grad, grad_input, r, c); });
}
int mimillm_sum_rows_f32(const float* a, float* out, std::int64_t r, std::int64_t c) {
    return guarded([&] { require(pointers({a, out}), "null pointer"); require(r >= 0 && c > 0, "invalid sum dimensions"); mimillm::sum_rows_f32(a, out, r, c); });
}
int mimillm_sum_rows_backward_f32(const float* grad, float* grad_input, std::int64_t r, std::int64_t c) {
    return guarded([&] { require(pointers({grad, grad_input}), "null pointer"); require(r >= 0 && c > 0, "invalid sum dimensions"); mimillm::sum_rows_backward_f32(grad, grad_input, r, c); });
}
int mimillm_relu_f32(const float* a, float* out, std::int64_t n) {
    return guarded([&] { require(pointers({a, out}), "null pointer"); require(n >= 0, "count must be non-negative"); mimillm::relu_f32(a, out, n); });
}
int mimillm_relu_backward_f32(const float* a, const float* grad, float* out, std::int64_t n) {
    return guarded([&] { require(pointers({a, grad, out}), "null pointer"); require(n >= 0, "count must be non-negative"); mimillm::relu_backward_f32(a, grad, out, n); });
}
int mimillm_embedding_gather_f32(const float* table, const std::int32_t* ids, float* out, std::int64_t vocab, std::int64_t width, std::int64_t count) {
    return guarded([&] { require(pointers({table, ids, out}), "null pointer"); require(vocab > 0 && width > 0 && count >= 0, "invalid embedding dimensions"); mimillm::embedding_gather_f32(table, ids, out, vocab, width, count); });
}
int mimillm_embedding_scatter_add_f32(const std::int32_t* ids, const float* grad, float* out, std::int64_t vocab, std::int64_t width, std::int64_t count) {
    return guarded([&] { require(pointers({ids, grad, out}), "null pointer"); require(vocab > 0 && width > 0 && count >= 0, "invalid embedding dimensions"); mimillm::embedding_scatter_add_f32(ids, grad, out, vocab, width, count); });
}
int mimillm_cross_entropy_f32(const float* logits, const std::int32_t* targets, float* loss, std::int64_t rows, std::int64_t classes) {
    return guarded([&] { require(pointers({logits, targets, loss}), "null pointer"); require(rows > 0 && classes > 0, "invalid cross entropy dimensions"); *loss = mimillm::cross_entropy_f32(logits, targets, rows, classes); });
}
int mimillm_cross_entropy_backward_f32(const float* logits, const std::int32_t* targets, float* grad, std::int64_t rows, std::int64_t classes) {
    return guarded([&] { require(pointers({logits, targets, grad}), "null pointer"); require(rows > 0 && classes > 0, "invalid cross entropy dimensions"); mimillm::cross_entropy_backward_f32(logits, targets, grad, rows, classes); });
}
int mimillm_adamw_f32(float* parameter, const float* gradient, float* first, float* second, std::int64_t count, float lr, float beta1, float beta2, float epsilon, float decay, std::int64_t step) {
    return guarded([&] { require(pointers({parameter, gradient, first, second}), "null pointer"); require(count >= 0 && lr > 0 && beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1 && epsilon > 0 && step > 0, "invalid AdamW arguments"); mimillm::adamw_f32(parameter, gradient, first, second, count, lr, beta1, beta2, epsilon, decay, step); });
}

}  // extern "C"
