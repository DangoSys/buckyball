#include "math/math.h"
#include <stdint.h>

extern "C" void rvv_tanh_launch(float *output, const float *input, size_t count,
                                uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t offset = 0; offset < count;) {
    const size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto bits = __riscv_vreinterpret_u32m1(values);
    auto magnitude = __riscv_vand_vx_u32m1(bits, 0x7fffffff, vl);
    auto finite = __riscv_vmsleu_vx_u32m1_b32(magnitude, 0x7f7fffff, vl);
    auto absolute =
        __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(0, vl),
                                 __riscv_vfabs_v_f32m1(values, vl), finite, vl);
    auto saturated = __riscv_vfmin_vf_f32m1(absolute, 10, vl);
    auto e = rvv_exp(__riscv_vfmul_vf_f32m1(saturated, -2, vl), vl);
    auto numerator = __riscv_vfrsub_vf_f32m1(e, 1, vl);
    auto denominator = __riscv_vfadd_vf_f32m1(e, 1, vl);
    auto result = __riscv_vfdiv_vv_f32m1(numerator, denominator, vl);
    // The odd series avoids exponential cancellation near the origin.
    auto bounded = __riscv_vfmin_vf_f32m1(absolute, 0.125f, vl);
    auto square = __riscv_vfmul_vv_f32m1(bounded, bounded, vl);
    auto polynomial = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vf_f32m1(square, -17.0f / 315.0f, vl), 2.0f / 15.0f, vl);
    polynomial = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(polynomial, square, vl), -1.0f / 3.0f, vl);
    polynomial = __riscv_vfmul_vv_f32m1(polynomial, square, vl);
    auto series = __riscv_vfadd_vv_f32m1(
        bounded, __riscv_vfmul_vv_f32m1(polynomial, bounded, vl), vl);
    auto small = __riscv_vmfle_vf_f32m1_b32(absolute, 0.125f, vl);
    result = __riscv_vmerge_vvm_f32m1(result, series, small, vl);
    auto infinity = __riscv_vmseq_vx_u32m1_b32(magnitude, 0x7f800000, vl);
    auto nan = __riscv_vmsgtu_vx_u32m1_b32(magnitude, 0x7f800000, vl);
    auto quiet =
        __riscv_vreinterpret_f32m1(__riscv_vor_vx_u32m1(bits, 0x00400000, vl));
    result = __riscv_vmerge_vvm_f32m1(result, __riscv_vfmv_v_f_f32m1(1, vl),
                                      infinity, vl);
    result = __riscv_vmerge_vvm_f32m1(result, quiet, nan, vl);
    auto signaling = __riscv_vmand_mm_b32(
        nan,
        __riscv_vmseq_vx_u32m1_b32(__riscv_vand_vx_u32m1(bits, 0x00400000, vl),
                                   0, vl),
        vl);
    if (__riscv_vcpop_m_b32(signaling, vl))
      asm volatile("csrs fflags, %0" ::"r"(uint32_t(16)) : "memory");
    result = __riscv_vfsgnj_vv_f32m1(result, values, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
