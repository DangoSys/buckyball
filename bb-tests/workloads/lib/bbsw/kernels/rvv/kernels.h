#pragma once
#include <stddef.h>
#include <stdint.h>

extern "C" {
void rvv_matmul(float *output, const float *lhs, const float *rhs, size_t rows,
                size_t cols, size_t inner);
void rvv_quant(uint8_t *output, const uint32_t *input, size_t count,
               uint32_t *status);
void rvv_silu(float *output, const float *input, size_t count);
void rvv_swiglu(float *output, const float *gate, const float *up,
                size_t count);
void rvv_snake(float *output, const float *input, size_t channels,
               size_t length, const float *log_alpha, const float *log_beta);
void rvv_norm(float *output, const float *input, const float *weight,
              size_t width, uint32_t meanMultiplierBits, uint32_t epsilonBits);
void rvv_softmax(float *output, const float *input, size_t rows, size_t width);
void rvv_rope(float *output, const float *input, const float *frequencies,
              const int64_t *positions, size_t rows, size_t headDimAndHeads);
}
