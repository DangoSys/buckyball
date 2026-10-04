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
  auto x = __riscv_vfwcvt_f_f_v_f64m2(input, vl);
  constexpr double invln2 = 0x1.71547652b82fep+5;
  auto kd = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(0x1.8p52, vl),
                                    invln2, x, vl);
  auto ki = __riscv_vreinterpret_u64m2(kd);
  kd = __riscv_vfsub_vf_f64m2(kd, 0x1.8p52, vl);
  auto r = __riscv_vfneg_v_f64m2(kd, vl);
  r = __riscv_vfmacc_vf_f64m2(r, invln2, x, vl);
  auto offset = __riscv_vand_vx_u64m2(ki, 31, vl);
  offset = __riscv_vsll_vx_u64m2(offset, 3, vl);
  auto indices = __riscv_vnsrl_wx_u32m1(offset, 0, vl);
  auto scale = __riscv_vluxei32_v_u64m2(exp_table, indices, vl);
  scale = __riscv_vadd_vv_u64m2(scale, __riscv_vsll_vx_u64m2(ki, 47, vl), vl);
  auto z =
      __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(0x1.ebfce50fac4f3p-13, vl),
                              0x1.c6af84b912394p-20, r, vl);
  auto y = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(1, vl),
                                   0x1.62e42ff0c52d6p-6, r, vl);
  auto r2 = __riscv_vfmul_vv_f64m2(r, r, vl);
  y = __riscv_vfmacc_vv_f64m2(y, z, r2, vl);
  y = __riscv_vfmul_vv_f64m2(y, __riscv_vreinterpret_f64m2(scale), vl);
  auto result = __riscv_vfncvt_f_f_w_f32m1(y, vl);
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
