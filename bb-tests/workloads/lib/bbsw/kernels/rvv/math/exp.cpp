// Vector adaptation of glibc 2.42 sysdeps/ieee754/flt-32/e_expf.c.
// Copyright (C) 2017-2025 Free Software Foundation, Inc.
// LGPL-2.1-or-later; see ../../../thirdparty/glibc/COPYING.LIB.
#include "constants.h"
#include "math.h"

extern "C" vfloat32m1_t rvv_exp(vfloat32m1_t value, size_t vl) {
  auto bits = __riscv_vreinterpret_u32m1(value);
  auto magnitude = __riscv_vand_vx_u32m1(bits, 0x7fffffff, vl);
  auto finite = __riscv_vmsltu_vx_u32m1_b32(magnitude, 0x7f800000, vl);
  auto ordered = __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(0, vl), value,
                                          finite, vl);
  auto normal = __riscv_vmfge_vf_f32m1_b32(ordered, -0x1.9d1d9ep6f, vl);
  normal = __riscv_vmand_mm_b32(
      normal, __riscv_vmfle_vf_f32m1_b32(ordered, 0x1.62e42ep6f, vl), vl);
  normal = __riscv_vmand_mm_b32(normal, finite, vl);
  auto input = __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(0, vl), value,
                                        normal, vl);
  auto scaled = __riscv_vfmul_vf_f32m1(input, 0x1.715476p5f, vl);
  auto kd = __riscv_vfadd_vf_f32m1(scaled, 0x1.8p23f, vl);
  kd = __riscv_vfsub_vf_f32m1(kd, 0x1.8p23f, vl);
  auto ki = __riscv_vfcvt_rtz_x_f_v_i32m1(kd, vl);
  auto r = __riscv_vfnmsac_vf_f32m1(input, 0x1.62e4p-6f, kd, vl);
  r = __riscv_vfnmsac_vf_f32m1(r, 0x1.7f7d1cp-25f, kd, vl);
  auto index = __riscv_vand_vx_u32m1(__riscv_vreinterpret_u32m1(ki), 31, vl);
  index = __riscv_vsll_vx_u32m1(index, 2, vl);
  auto table = __riscv_vluxei32_v_f32m1(exp_table, index, vl);
  auto exponent = __riscv_vsra_vx_i32m1(ki, 5, vl);
  auto bounded = __riscv_vmax_vx_i32m1(exponent, -126, vl);
  bounded = __riscv_vmin_vx_i32m1(bounded, 127, vl);
  auto tail = __riscv_vsub_vv_i32m1(exponent, bounded, vl);
  auto scale_bits =
      __riscv_vsll_vx_i32m1(__riscv_vadd_vx_i32m1(bounded, 127, vl), 23, vl);
  auto tail_bits =
      __riscv_vsll_vx_i32m1(__riscv_vadd_vx_i32m1(tail, 127, vl), 23, vl);
  auto polynomial = __riscv_vfmacc_vf_f32m1(__riscv_vfmv_v_f_f32m1(0.5f, vl),
                                            1.0f / 6.0f, r, vl);
  auto y = __riscv_vfadd_vf_f32m1(r, 1.0f, vl);
  y = __riscv_vfmacc_vv_f32m1(y, polynomial, __riscv_vfmul_vv_f32m1(r, r, vl),
                              vl);
  y = __riscv_vfmul_vv_f32m1(y, table, vl);
  y = __riscv_vfmul_vv_f32m1(
      y, __riscv_vreinterpret_f32m1(__riscv_vreinterpret_u32m1(tail_bits)), vl);
  auto result = __riscv_vfmul_vv_f32m1(
      y, __riscv_vreinterpret_f32m1(__riscv_vreinterpret_u32m1(scale_bits)),
      vl);
  auto under = __riscv_vmflt_vf_f32m1_b32(ordered, -0x1.9d1d9ep6f, vl);
  under = __riscv_vmand_mm_b32(under, finite, vl);
  auto helper = __riscv_vfmv_v_f_f32m1(0x1.4p-75f, vl);
  helper = __riscv_vfmul_vv_f32m1_m(under, helper, helper, vl);
  result = __riscv_vmerge_vvm_f32m1(result, helper, under, vl);
  auto zero = __riscv_vmflt_vf_f32m1_b32(ordered, -0x1.9fe368p6f, vl);
  result = __riscv_vfmerge_vfm_f32m1(result, 0, zero, vl);
  auto over = __riscv_vmfgt_vf_f32m1_b32(ordered, 0x1.62e42ep6f, vl);
  auto overflow = __riscv_vmand_mm_b32(over, finite, vl);
  helper = __riscv_vfmv_v_f_f32m1(0x1p97f, vl);
  helper = __riscv_vfmul_vv_f32m1_m(overflow, helper, helper, vl);
  result = __riscv_vmerge_vvm_f32m1(result, helper, overflow, vl);
  auto infinity = __riscv_vmseq_vx_u32m1_b32(bits, 0x7f800000, vl);
  result = __riscv_vfmerge_vfm_f32m1(result, __builtin_inff(), infinity, vl);
  auto negative_infinity = __riscv_vmseq_vx_u32m1_b32(bits, 0xff800000, vl);
  result = __riscv_vfmerge_vfm_f32m1(result, 0, negative_infinity, vl);
  auto nan = __riscv_vmsgtu_vx_u32m1_b32(magnitude, 0x7f800000, vl);
  return __riscv_vmerge_vvm_f32m1(
      result, __riscv_vfadd_vv_f32m1_m(nan, value, value, vl), nan, vl);
}
