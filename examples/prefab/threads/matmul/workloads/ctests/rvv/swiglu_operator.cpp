// SwiGLU on the GP domain: output = gate / (1 + exp(-gate)) * up.
#include <rvv/math.h>

extern "C" void swiglu(float *output, const float *gate, const float *up,
                       size_t count, uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t offset = 0; offset < count;) {
    const size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto value = __riscv_vle32_v_f32m1(gate + offset, vl);
    auto other = __riscv_vle32_v_f32m1(up + offset, vl);
    auto denominator = __riscv_vfadd_vf_f32m1(
        rvv_exp(__riscv_vfneg_v_f32m1(value, vl), vl), 1, vl);
    auto result = __riscv_vfmul_vv_f32m1(
        __riscv_vfdiv_vv_f32m1(value, denominator, vl), other, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
