// RMSNorm of one row on the GP domain:
// output = weight * x / sqrt(sum(x^2) * mean + epsilon), ordered reduction.
#include <riscv_vector.h>
#include <stdint.h>

extern "C" void norm(float *output, const float *input, const float *weight,
                     size_t width, uint32_t meanBits, uint32_t epsilonBits,
                     uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  auto sum = __riscv_vfmv_v_f_f32m1(0, 1);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto squares = __riscv_vfmul_vv_f32m1(values, values, vl);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(squares, sum, vl);
    offset += vl;
  }
  auto mean = __riscv_vfmul_vv_f32m1(
      sum, __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(meanBits, 1)), 1);
  auto biased = __riscv_vfadd_vv_f32m1(
      mean, __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(epsilonBits, 1)),
      1);
  auto reciprocal =
      __riscv_vfrdiv_vf_f32m1(__riscv_vfsqrt_v_f32m1(biased, 1), 1, 1);
  const float scale = __riscv_vfmv_f_s_f32m1_f32(reciprocal);
  for (size_t offset = 0; offset < width;) {
    const size_t vl = __riscv_vsetvl_e32m1(width - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto scaled = __riscv_vfmul_vf_f32m1(values, scale, vl);
    auto weights = __riscv_vle32_v_f32m1(weight + offset, vl);
    __riscv_vse32_v_f32m1(output + offset,
                          __riscv_vfmul_vv_f32m1(weights, scaled, vl), vl);
    offset += vl;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
