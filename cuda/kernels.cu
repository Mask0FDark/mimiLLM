#include <cuda_runtime.h>

// NVRTC does not ship the host C++ standard library. These fixed-width aliases
// match the 64-bit CUDA platforms supported by mimiLLM.
namespace std {
using int64_t = long long;
using int32_t = int;
}

constexpr int tile = 16;

extern "C" __global__ void mimillm_add(
    const float* left, const float* right, float* output, std::int64_t count
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = left[index] + right[index];
}

extern "C" __global__ void mimillm_multiply(
    const float* left, const float* right, float* output, std::int64_t count
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = left[index] * right[index];
}

extern "C" __global__ void mimillm_scalar_multiply(
    const float* input, float scalar, float* output, std::int64_t count
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scalar;
}

extern "C" __global__ void mimillm_permute(
    const float* input, float* output, const std::int64_t* source_strides,
    const std::int64_t* output_shape, const std::int64_t* output_strides,
    const std::int64_t* axes, std::int64_t dimensions, std::int64_t count
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    std::int64_t source_index = 0;
    for (std::int64_t axis = 0; axis < dimensions; ++axis) {
        const auto coordinate = (flat / output_strides[axis]) % output_shape[axis];
        source_index += coordinate * source_strides[axes[axis]];
    }
    output[flat] = input[source_index];
}

__device__ std::int64_t mimillm_broadcast_index(
    std::int64_t flat, const std::int64_t* operand_strides,
    const std::int64_t* output_shape, const std::int64_t* output_strides,
    std::int64_t dimensions
) {
    std::int64_t result = 0;
    for (std::int64_t axis = 0; axis < dimensions; ++axis) {
        if (operand_strides[axis]) {
            result += ((flat / output_strides[axis]) % output_shape[axis]) * operand_strides[axis];
        }
    }
    return result;
}

extern "C" __global__ void mimillm_broadcast_binary(
    const float* left, const float* right, float* output,
    const std::int64_t* left_strides, const std::int64_t* right_strides,
    const std::int64_t* output_shape, const std::int64_t* output_strides,
    std::int64_t dimensions, std::int64_t count, std::int32_t operation
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const auto left_index = mimillm_broadcast_index(flat, left_strides, output_shape, output_strides, dimensions);
    const auto right_index = mimillm_broadcast_index(flat, right_strides, output_shape, output_strides, dimensions);
    const float a = left[left_index];
    const float b = right[right_index];
    if (operation == 0) output[flat] = a + b;
    else if (operation == 1) output[flat] = a - b;
    else if (operation == 2) output[flat] = a * b;
    else output[flat] = a / b;
}

extern "C" __global__ void mimillm_broadcast_backward(
    const float* left, const float* right, const float* grad_output,
    float* grad_left, float* grad_right,
    const std::int64_t* left_strides, const std::int64_t* right_strides,
    const std::int64_t* output_shape, const std::int64_t* output_strides,
    std::int64_t dimensions, std::int64_t count, std::int32_t operation
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const auto left_index = mimillm_broadcast_index(flat, left_strides, output_shape, output_strides, dimensions);
    const auto right_index = mimillm_broadcast_index(flat, right_strides, output_shape, output_strides, dimensions);
    const float a = left[left_index];
    const float b = right[right_index];
    const float upstream = grad_output[flat];
    float da = upstream;
    float db = upstream;
    if (operation == 1) db = -upstream;
    else if (operation == 2) { da = upstream * b; db = upstream * a; }
    else if (operation == 3) { da = upstream / b; db = -upstream * a / (b * b); }
    atomicAdd(grad_left + left_index, da);
    atomicAdd(grad_right + right_index, db);
}

extern "C" __global__ void mimillm_matmul(
    const float* left, const float* right, float* output,
    std::int64_t batches, std::int64_t rows, std::int64_t inner, std::int64_t columns
) {
    __shared__ float left_tile[tile][tile];
    __shared__ float right_tile[tile][tile];
    const auto batch = static_cast<std::int64_t>(blockIdx.z);
    const auto row = static_cast<std::int64_t>(blockIdx.y) * tile + threadIdx.y;
    const auto column = static_cast<std::int64_t>(blockIdx.x) * tile + threadIdx.x;
    const auto left_offset = batch * rows * inner;
    const auto right_offset = batch * inner * columns;
    float total = 0.0F;
    for (std::int64_t start = 0; start < inner; start += tile) {
        const auto left_column = start + threadIdx.x;
        const auto right_row = start + threadIdx.y;
        left_tile[threadIdx.y][threadIdx.x] = row < rows && left_column < inner
            ? left[left_offset + row * inner + left_column] : 0.0F;
        right_tile[threadIdx.y][threadIdx.x] = right_row < inner && column < columns
            ? right[right_offset + right_row * columns + column] : 0.0F;
        __syncthreads();
        for (int index = 0; index < tile; ++index) total += left_tile[threadIdx.y][index] * right_tile[index][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && column < columns) output[batch * rows * columns + row * columns + column] = total;
}

extern "C" __global__ void mimillm_softmax_rows(
    const float* input, float* output, std::int64_t rows, std::int64_t columns
) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    extern __shared__ float shared[];
    float local_max = -3.402823466e+38F;
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        local_max = fmaxf(local_max, input[row * columns + column]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = shared[0];
    float local_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        local_sum += expf(input[row * columns + column] - maximum);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = shared[0];
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        output[row * columns + column] = expf(input[row * columns + column] - maximum) / denominator;
    }
}

extern "C" __global__ void mimillm_softmax_backward(
    const float* output, const float* grad_output, float* grad_input,
    std::int64_t rows, std::int64_t columns
) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    extern __shared__ float shared[];
    float local_dot = 0.0F;
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const auto index = row * columns + column;
        local_dot += output[index] * grad_output[index];
    }
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float dot = shared[0];
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const auto index = row * columns + column;
        grad_input[index] = output[index] * (grad_output[index] - dot);
    }
}

extern "C" __global__ void mimillm_sum_rows(
    const float* input, float* output, std::int64_t rows, std::int64_t columns
) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    extern __shared__ float shared[];
    float local_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) local_sum += input[row * columns + column];
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = shared[0];
}

extern "C" __global__ void mimillm_sum_rows_backward(
    const float* gradient, float* output, std::int64_t count, std::int64_t columns
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = gradient[index / columns];
}

extern "C" __global__ void mimillm_relu(const float* input, float* output, std::int64_t count) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = fmaxf(input[index], 0.0F);
}

extern "C" __global__ void mimillm_relu_backward(
    const float* input, const float* gradient, float* output, std::int64_t count
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] > 0.0F ? gradient[index] : 0.0F;
}

extern "C" __global__ void mimillm_embedding_gather(
    const float* table, const std::int32_t* indices, float* output,
    std::int64_t width, std::int64_t output_count
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < output_count) output[flat] = table[static_cast<std::int64_t>(indices[flat / width]) * width + flat % width];
}

extern "C" __global__ void mimillm_embedding_scatter(
    const std::int32_t* indices, const float* gradient, float* output,
    std::int64_t width, std::int64_t count
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < count) atomicAdd(output + static_cast<std::int64_t>(indices[flat / width]) * width + flat % width, gradient[flat]);
}

extern "C" __global__ void mimillm_cross_entropy_loss(
    const float* logits, const std::int32_t* targets, const float* weights,
    float weight_sum, float* loss, std::int64_t rows, std::int64_t classes
) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const float weight = weights ? weights[row] : 1.0F;
    if (weight == 0.0F) return;
    extern __shared__ float shared[];
    float local_max = -3.402823466e+38F;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) local_max = fmaxf(local_max, logits[row * classes + column]);
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = shared[0];
    float local_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) local_sum += expf(logits[row * classes + column] - maximum);
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float value = maximum + logf(shared[0]) - logits[row * classes + targets[row]];
        atomicAdd(loss, weight * value / weight_sum);
    }
}

extern "C" __global__ void mimillm_cross_entropy_gradient(
    float* probabilities, const std::int32_t* targets, const float* weights,
    float weight_sum, std::int64_t count, std::int64_t classes
) {
    const auto flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const auto row = flat / classes;
    const auto column = flat % classes;
    const float weight = weights ? weights[row] : 1.0F;
    probabilities[flat] = (probabilities[flat] - (column == targets[row] ? 1.0F : 0.0F)) * weight / weight_sum;
}

extern "C" __global__ void mimillm_adamw(
    float* parameter, const float* gradient, float* first, float* second,
    std::int64_t count, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float correction1, float correction2
) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = gradient[index];
    const float first_value = beta1 * first[index] + (1.0F - beta1) * value;
    const float second_value = beta2 * second[index] + (1.0F - beta2) * value * value;
    first[index] = first_value;
    second[index] = second_value;
    const float update = (first_value / correction1) / (sqrtf(second_value / correction2) + epsilon) + weight_decay * parameter[index];
    parameter[index] -= learning_rate * update;
}

extern "C" __global__ void mimillm_sum_squares(const float* input, float* output, std::int64_t count) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(output, input[index] * input[index]);
}
