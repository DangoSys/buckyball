#include "kernels.h"
#include <riscv_vector.h>

extern "C" void rvv_layernorm(float *output, const float *input,
                              const float *weight, const float *bias,
                              uint64_t widthAndBias, uint64_t parameters,
                              uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  const bool hasBias = widthAndBias >> 32;
  const size_t width = uint32_t(widthAndBias);
  const float multiplier = __builtin_bit_cast(float, uint32_t(parameters));
  const float epsilon = __builtin_bit_cast(float, uint32_t(parameters >> 32));
  auto sum = __riscv_vfmv_v_f_f32m1(0, 1);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(values, sum, vl);
    offset += vl;
  }
  const float mean = __riscv_vfmv_f_s_f32m1_f32(sum) * multiplier;
  sum = __riscv_vfmv_v_f_f32m1(0, 1);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto centered = __riscv_vfsub_vf_f32m1(values, mean, vl);
    auto squares = __riscv_vfmul_vv_f32m1(centered, centered, vl);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(squares, sum, vl);
    offset += vl;
  }
  const float variance = __riscv_vfmv_f_s_f32m1_f32(sum) * multiplier;
  const float biased = variance + epsilon;
  float root;
  asm volatile("fsqrt.s %0, %1" : "=f"(root) : "f"(biased));
  const float scale = 1.0f / root;
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto centered = __riscv_vfsub_vf_f32m1(values, mean, vl);
    auto normalized = __riscv_vfmul_vf_f32m1(centered, scale, vl);
    auto weights = __riscv_vle32_v_f32m1(weight + offset, vl);
    auto weighted = __riscv_vfmul_vv_f32m1(normalized, weights, vl);
    if (hasBias) {
      auto biases = __riscv_vle32_v_f32m1(bias + offset, vl);
      weighted = __riscv_vfadd_vv_f32m1(weighted, biases, vl);
    }
    __riscv_vse32_v_f32m1(output + offset, weighted, vl);
    offset += vl;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
