#include "kernels.h"
#include <riscv_vector.h>

template <bool weighted>
static void norm(float *output, const float *input, const float *weight,
                 size_t width, uint32_t meanMultiplierBits,
                 uint32_t epsilonBits) {
  auto sum = __riscv_vfmv_v_f_f32m1(0, 1);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto squares = __riscv_vfmul_vv_f32m1(values, values, vl);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(squares, sum, vl);
    offset += vl;
  }
  auto meanMultiplier =
      __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(meanMultiplierBits, 1));
  auto epsilon =
      __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(epsilonBits, 1));
  auto mean = __riscv_vfmul_vv_f32m1(sum, meanMultiplier, 1);
  auto biased = __riscv_vfadd_vv_f32m1(mean, epsilon, 1);
  auto root = __riscv_vfsqrt_v_f32m1(biased, 1);
  auto reciprocal = __riscv_vfrdiv_vf_f32m1(root, 1, 1);
  const float scale = __riscv_vfmv_f_s_f32m1_f32(reciprocal);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto scaled = __riscv_vfmul_vf_f32m1(values, scale, vl);
    if constexpr (weighted) {
      auto weights = __riscv_vle32_v_f32m1(weight + offset, vl);
      scaled = __riscv_vfmul_vv_f32m1(weights, scaled, vl);
    }
    __riscv_vse32_v_f32m1(output + offset, scaled, vl);
    offset += vl;
  }
}

extern "C" void rvv_norm(float *output, const float *input, const float *weight,
                         size_t width, uint32_t mean, uint32_t epsilon) {
  norm<true>(output, input, weight, width, mean, epsilon);
}

extern "C" void rvv_norm_no_weight(float *output, const float *input,
                                   size_t width, uint32_t mean,
                                   uint32_t epsilon) {
  norm<false>(output, input, nullptr, width, mean, epsilon);
}

extern "C" void rvv_norm_launch(float *output, const float *input,
                                const float *weight, size_t width,
                                uint32_t mean, uint32_t epsilon,
                                uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_norm(output, input, weight, width, mean, epsilon);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}

extern "C" void rvv_norm_no_weight_launch(float *output, const float *input,
                                          size_t width, uint32_t mean,
                                          uint32_t epsilon, uint32_t *flags,
                                          uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_norm_no_weight(output, input, width, mean, epsilon);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
