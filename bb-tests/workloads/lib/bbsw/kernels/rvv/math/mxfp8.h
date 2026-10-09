#pragma once
#include <riscv_vector.h>
#include <stdint.h>

static inline vfloat32m2_t rvv_mxfp8_load16(const uint8_t *input,
                                            unsigned scale, uint32_t &status) {
  constexpr size_t vl = 16;
  if (scale == 255) {
    status = 1U << 8;
    return __riscv_vfmv_v_f_f32m2(0, vl);
  }
  auto bytes = __riscv_vle8_v_u8mf2(input, vl);
  auto code = __riscv_vzext_vf2_u32m2(__riscv_vzext_vf2_u16m1(bytes, vl), vl);
  auto sign =
      __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(code, 128, vl), 24, vl);
  auto magnitude = __riscv_vand_vx_u32m2(code, 127, vl);
  if (__riscv_vcpop_m_b16(__riscv_vmseq_vx_u32m2_b16(magnitude, 127, vl), vl)) {
    status = 1U << 8;
    return __riscv_vfmv_v_f_f32m2(0, vl);
  }
  auto exponent = __riscv_vsrl_vx_u32m2(magnitude, 3, vl);
  auto fraction = __riscv_vand_vx_u32m2(code, 7, vl);
  auto subnormal = __riscv_vmseq_vx_u32m2_b16(exponent, 0, vl);
  auto highest = __riscv_vmv_v_x_u32m2(0, vl);
  highest = __riscv_vmerge_vxm_u32m2(
      highest, 1, __riscv_vmsgeu_vx_u32m2_b16(fraction, 2, vl), vl);
  highest = __riscv_vmerge_vxm_u32m2(
      highest, 2, __riscv_vmsgeu_vx_u32m2_b16(fraction, 4, vl), vl);
  auto normalExponent = __riscv_vadd_vx_i32m2(
      __riscv_vreinterpret_v_u32m2_i32m2(exponent), int(scale) - 7, vl);
  auto subExponent = __riscv_vadd_vx_i32m2(
      __riscv_vreinterpret_v_u32m2_i32m2(highest), int(scale) - 9, vl);
  auto resultExponent =
      __riscv_vmerge_vvm_i32m2(normalExponent, subExponent, subnormal, vl);
  auto overflow =
      __riscv_vmand_mm_b16(__riscv_vmsge_vx_i32m2_b16(resultExponent, 255, vl),
                           __riscv_vmsne_vx_u32m2_b16(magnitude, 0, vl), vl);
  if (__riscv_vcpop_m_b16(overflow, vl)) {
    status = 2U << 8;
    return __riscv_vfmv_v_f_f32m2(0, vl);
  }
  auto normalSignificand =
      __riscv_vsll_vx_u32m2(__riscv_vor_vx_u32m2(fraction, 8, vl), 20, vl);
  auto subSignificand = __riscv_vsll_vv_u32m2(
      fraction, __riscv_vrsub_vx_u32m2(highest, 23, vl), vl);
  auto significand = __riscv_vmerge_vvm_u32m2(normalSignificand, subSignificand,
                                              subnormal, vl);
  auto normalBits = __riscv_vor_vv_u32m2(
      __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(resultExponent),
                            23, vl),
      __riscv_vand_vx_u32m2(significand, 0x7fffff, vl), vl);
  auto tinyBits =
      __riscv_vsrl_vv_u32m2(significand,
                            __riscv_vreinterpret_v_i32m2_u32m2(
                                __riscv_vrsub_vx_i32m2(resultExponent, 1, vl)),
                            vl);
  auto bits = __riscv_vmerge_vvm_u32m2(
      normalBits, tinyBits, __riscv_vmsle_vx_i32m2_b16(resultExponent, 0, vl),
      vl);
  bits = __riscv_vmerge_vxm_u32m2(
      bits, 0, __riscv_vmseq_vx_u32m2_b16(magnitude, 0, vl), vl);
  return __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vor_vv_u32m2(bits, sign, vl));
}
