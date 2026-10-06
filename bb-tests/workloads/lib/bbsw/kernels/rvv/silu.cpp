#include "kernels.h"
#include "math/math.h"

extern "C" void rvv_silu(float *output, const float *input, size_t count) {
  for (size_t offset = 0; offset < count;) {
    const size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto value = __riscv_vle32_v_f32m1(input + offset, vl);
    auto denominator = __riscv_vfadd_vf_f32m1(
        rvv_exp(__riscv_vfneg_v_f32m1(value, vl), vl), 1, vl);
    auto sigmoid = __riscv_vfrdiv_vf_f32m1(denominator, 1, vl);
    auto result = __riscv_vfmul_vv_f32m1(value, sigmoid, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
}

extern "C" void rvv_swiglu(float *output, const float *gate, const float *up,
                           size_t count) {
  for (size_t offset = 0; offset < count;) {
    const size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto value = __riscv_vle32_v_f32m1(gate + offset, vl);
    auto other = __riscv_vle32_v_f32m1(up + offset, vl);
    auto denominator = __riscv_vfadd_vf_f32m1(
        rvv_exp(__riscv_vfneg_v_f32m1(value, vl), vl), 1, vl);
    auto sigmoid = __riscv_vfrdiv_vf_f32m1(denominator, 1, vl);
    auto result = __riscv_vfmul_vv_f32m1(
        __riscv_vfmul_vv_f32m1(value, sigmoid, vl), other, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
}

extern "C" void rvv_silu_launch(float *output, const float *input, size_t count,
                                uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_silu(output, input, count);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}

extern "C" void rvv_swiglu_launch(float *output, const float *gate,
                                  const float *up, size_t count,
                                  uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_swiglu(output, gate, up, count);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
