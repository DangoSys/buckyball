#include "kernels.h"
#include "math/math.h"

extern "C" void rvv_snake(float *output, const float *input, size_t channels,
                          size_t length, const float *log_alpha,
                          const float *log_beta) {
  for (size_t channel = 0; channel < channels; ++channel) {
    auto alpha = rvv_exp(__riscv_vfmv_v_f_f32m1(log_alpha[channel], 1), 1);
    auto beta = __riscv_vfadd_vf_f32m1(
        rvv_exp(__riscv_vfmv_v_f_f32m1(log_beta[channel], 1), 1), 1e-9f, 1);
    auto reciprocal = __riscv_vfrdiv_vf_f32m1(beta, 1, 1);
    const float scale = __riscv_vfmv_f_s_f32m1_f32(alpha);
    const float inverse = __riscv_vfmv_f_s_f32m1_f32(reciprocal);
    for (size_t offset = 0; offset < length;) {
      const size_t vl = __riscv_vsetvl_e32m1(length - offset);
      const size_t index = channel * length + offset;
      auto value = __riscv_vle32_v_f32m1(input + index, vl);
      auto angle = __riscv_vfmul_vf_f32m1(value, scale, vl);
      auto sine = rvv_sin(angle, vl);
      auto square = __riscv_vfmul_vv_f32m1(sine, sine, vl);
      auto scaled = __riscv_vfmul_vf_f32m1(square, inverse, vl);
      auto result = __riscv_vfadd_vv_f32m1(value, scaled, vl);
      __riscv_vse32_v_f32m1(output + index, result, vl);
      offset += vl;
    }
  }
}
