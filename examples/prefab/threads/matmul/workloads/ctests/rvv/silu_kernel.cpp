// SiLU on the GP domain: output = x / (1 + exp(-x)).
#include <rvv/math.h>

extern "C" void silu(float *output, const float *input, size_t count,
                     uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t offset = 0; offset < count;) {
    const size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto value = __riscv_vle32_v_f32m1(input + offset, vl);
    auto denominator = __riscv_vfadd_vf_f32m1(
        rvv_exp(__riscv_vfneg_v_f32m1(value, vl), vl), 1, vl);
    auto result = __riscv_vfdiv_vv_f32m1(value, denominator, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
