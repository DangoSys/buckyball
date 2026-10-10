#include "kernels.h"
#include <riscv_vector.h>

extern "C" void rvv_dequant(uint32_t *output, const uint8_t *input,
                            const uint8_t *scales, size_t count,
                            uint32_t *status) {
  *status = 0;
  for (size_t block = 0; block < count / 32; ++block) {
    unsigned scale = scales[block];
    if (scale == 255) {
      *status = 1;
      return;
    }
    for (size_t begin = 0; begin < 32;) {
      size_t vl = __riscv_vsetvl_e32m1(32 - begin);
      auto bytes = __riscv_vle8_v_u8mf4(input + block * 32 + begin, vl);
      auto code =
          __riscv_vzext_vf2_u32m1(__riscv_vzext_vf2_u16mf2(bytes, vl), vl);
      auto sign =
          __riscv_vsll_vx_u32m1(__riscv_vand_vx_u32m1(code, 128, vl), 24, vl);
      auto magnitude = __riscv_vand_vx_u32m1(code, 127, vl);
      if (__riscv_vcpop_m_b32(__riscv_vmseq_vx_u32m1_b32(magnitude, 127, vl),
                              vl)) {
        *status = 1;
        return;
      }
      auto exponent = __riscv_vsrl_vx_u32m1(magnitude, 3, vl);
      auto fraction = __riscv_vand_vx_u32m1(code, 7, vl);
      auto subnormal = __riscv_vmseq_vx_u32m1_b32(exponent, 0, vl);
      auto highest = __riscv_vmv_v_x_u32m1(0, vl);
      highest = __riscv_vmerge_vxm_u32m1(
          highest, 1, __riscv_vmsgeu_vx_u32m1_b32(fraction, 2, vl), vl);
      highest = __riscv_vmerge_vxm_u32m1(
          highest, 2, __riscv_vmsgeu_vx_u32m1_b32(fraction, 4, vl), vl);
      auto normalExponent = __riscv_vadd_vx_i32m1(
          __riscv_vreinterpret_v_u32m1_i32m1(exponent), int(scale) - 7, vl);
      auto subExponent = __riscv_vadd_vx_i32m1(
          __riscv_vreinterpret_v_u32m1_i32m1(highest), int(scale) - 9, vl);
      auto resultExponent =
          __riscv_vmerge_vvm_i32m1(normalExponent, subExponent, subnormal, vl);
      auto overflow = __riscv_vmand_mm_b32(
          __riscv_vmsge_vx_i32m1_b32(resultExponent, 255, vl),
          __riscv_vmsne_vx_u32m1_b32(magnitude, 0, vl), vl);
      if (__riscv_vcpop_m_b32(overflow, vl)) {
        *status = 2;
        return;
      }
      auto normalSignificand =
          __riscv_vsll_vx_u32m1(__riscv_vor_vx_u32m1(fraction, 8, vl), 20, vl);
      auto subSignificand = __riscv_vsll_vv_u32m1(
          fraction, __riscv_vrsub_vx_u32m1(highest, 23, vl), vl);
      auto significand = __riscv_vmerge_vvm_u32m1(
          normalSignificand, subSignificand, subnormal, vl);
      auto normalBits = __riscv_vor_vv_u32m1(
          __riscv_vsll_vx_u32m1(
              __riscv_vreinterpret_v_i32m1_u32m1(resultExponent), 23, vl),
          __riscv_vand_vx_u32m1(significand, 0x7fffff, vl), vl);
      auto tinyBits = __riscv_vsrl_vv_u32m1(
          significand,
          __riscv_vreinterpret_v_i32m1_u32m1(
              __riscv_vrsub_vx_i32m1(resultExponent, 1, vl)),
          vl);
      auto bits = __riscv_vmerge_vvm_u32m1(
          normalBits, tinyBits,
          __riscv_vmsle_vx_i32m1_b32(resultExponent, 0, vl), vl);
      bits = __riscv_vmerge_vxm_u32m1(
          bits, 0, __riscv_vmseq_vx_u32m1_b32(magnitude, 0, vl), vl);
      __riscv_vse32_v_u32m1(output + block * 32 + begin,
                            __riscv_vor_vv_u32m1(bits, sign, vl), vl);
      begin += vl;
    }
  }
}
