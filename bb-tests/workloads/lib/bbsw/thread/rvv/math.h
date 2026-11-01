// RVV exp and sin for the GP domain's test kernels: vector adaptations of
// glibc 2.42 sysdeps/ieee754/flt-32/e_expf.c and s_sinf.c
// (s_sincosf.h/sincosf_poly.h).
// Copyright (C) 2017-2025 Free Software Foundation, Inc.
// LGPL-2.1-or-later.
#pragma once
#include <riscv_vector.h>
#include <stdint.h>
static const float exp_table[32] = {
    0x1.0000000000000p+0f, 0x1.059b0e0000000p+0f, 0x1.0b55860000000p+0f,
    0x1.11301e0000000p+0f, 0x1.172b840000000p+0f, 0x1.1d48740000000p+0f,
    0x1.2387a60000000p+0f, 0x1.29e9e00000000p+0f, 0x1.306fe00000000p+0f,
    0x1.371a740000000p+0f, 0x1.3dea640000000p+0f, 0x1.44e0860000000p+0f,
    0x1.4bfdae0000000p+0f, 0x1.5342b60000000p+0f, 0x1.5ab07e0000000p+0f,
    0x1.6247ec0000000p+0f, 0x1.6a09e60000000p+0f, 0x1.71f75e0000000p+0f,
    0x1.7a11480000000p+0f, 0x1.82589a0000000p+0f, 0x1.8ace540000000p+0f,
    0x1.93737c0000000p+0f, 0x1.9c49180000000p+0f, 0x1.a5503c0000000p+0f,
    0x1.ae89fa0000000p+0f, 0x1.b7f7700000000p+0f, 0x1.c199be0000000p+0f,
    0x1.cb720e0000000p+0f, 0x1.d5818e0000000p+0f, 0x1.dfc9740000000p+0f,
    0x1.ea4afa0000000p+0f, 0x1.f507660000000p+0f,
};
static const uint32_t inv_pio4[24] = {

    0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};

static inline vfloat32m1_t rvv_exp(vfloat32m1_t value, size_t vl) {
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

static inline vfloat32m1_t rvv_sin(vfloat32m1_t value, size_t vl) {
  auto bits = __riscv_vreinterpret_u32m1(value);
  auto top =
      __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(bits, 20, vl), 0x7ff, vl);
  auto tiny = __riscv_vmsltu_vx_u32m1_b32(top, 0x398, vl);
  auto fast = __riscv_vmsltu_vx_u32m1_b32(top, 0x42f, vl);
  fast = __riscv_vmandn_mm_b32(fast, tiny, vl);
  auto finite = __riscv_vmsltu_vx_u32m1_b32(top, 0x7f8, vl);
  auto input =
      __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(0, vl), value, fast, vl);
  auto x = input;
  auto r = __riscv_vfmul_vf_f32m1(x, 0x1.45f306dc9c883p23, vl);
  auto n = __riscv_vfcvt_rtz_x_f_v_i32m1(r, vl);
  n = __riscv_vsra_vx_i32m1(__riscv_vadd_vx_i32m1(n, 0x800000, vl), 24, vl);
  auto small = __riscv_vmsltu_vx_u32m1_b32(top, 0x3f4, vl);
  n = __riscv_vmerge_vxm_i32m1(n, 0, small, vl);
  auto nd = __riscv_vfcvt_f_x_v_f32m1(n, vl);
  x = __riscv_vfnmsac_vf_f32m1(x, 0x1.921fb4p0f, nd, vl);
  x = __riscv_vfnmsac_vf_f32m1(x, 0x1.4442d2p-24f, nd, vl);
  auto quadrant = n;
  auto large = __riscv_vmandn_mm_b32(finite, fast, vl);
  large = __riscv_vmandn_mm_b32(large, tiny, vl);
  if (__riscv_vcpop_m_b32(large, vl)) {
    auto index =
        __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(bits, 26, vl), 15, vl);
    index = __riscv_vsll_vx_u32m1(index, 2, vl);
    auto a0 = __riscv_vluxei32_v_u32m1(inv_pio4, index, vl);
    auto a1 = __riscv_vluxei32_v_u32m1(inv_pio4 + 4, index, vl);
    auto a2 = __riscv_vluxei32_v_u32m1(inv_pio4 + 8, index, vl);
    auto shift =
        __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(bits, 23, vl), 7, vl);
    auto xi = __riscv_vor_vx_u32m1(__riscv_vand_vx_u32m1(bits, 0xffffff, vl),
                                   0x800000, vl);
    xi = __riscv_vsll_vv_u32m1(xi, shift, vl);
    auto low0 = __riscv_vmulhu_vv_u32m1(xi, a2, vl);
    auto low1 = __riscv_vmul_vv_u32m1(xi, a1, vl);
    auto low = __riscv_vadd_vv_u32m1(low0, low1, vl);
    auto carry_mask = __riscv_vmsltu_vv_u32m1_b32(low, low1, vl);
    auto high = __riscv_vadd_vv_u32m1(__riscv_vmul_vv_u32m1(xi, a0, vl),
                                      __riscv_vmulhu_vv_u32m1(xi, a1, vl), vl);
    auto carry = __riscv_vmerge_vxm_u32m1(__riscv_vmv_v_x_u32m1(0, vl), 1,
                                          carry_mask, vl);
    high = __riscv_vadd_vv_u32m1(high, carry, vl);
    auto q = __riscv_vsrl_vx_u32m1(__riscv_vadd_vx_u32m1(high, 0x20000000, vl),
                                   30, vl);
    high = __riscv_vsub_vv_u32m1(high, __riscv_vsll_vx_u32m1(q, 30, vl), vl);
    auto reduced =
        __riscv_vfcvt_f_x_v_f32m1(__riscv_vreinterpret_i32m1(high), vl);
    reduced = __riscv_vfmul_vf_f32m1(reduced, 0x1.921fb4p-30f, vl);
    auto fraction = __riscv_vfcvt_f_xu_v_f32m1(low, vl);
    reduced = __riscv_vfmacc_vf_f32m1(reduced, 0x1.921fb4p-62f, fraction, vl);
    auto q32 = __riscv_vreinterpret_i32m1(q);
    n = __riscv_vmerge_vvm_i32m1(n, q32, large, vl);
    auto sign = __riscv_vreinterpret_i32m1(__riscv_vsrl_vx_u32m1(bits, 31, vl));
    quadrant = __riscv_vmerge_vvm_i32m1(
        quadrant, __riscv_vadd_vv_i32m1(q32, sign, vl), large, vl);
    x = __riscv_vmerge_vvm_f32m1(x, reduced, large, vl);
  }
  auto x2 = __riscv_vfmul_vv_f32m1(x, x, vl);
  auto negative =
      __riscv_vmsne_vx_i32m1_b32(__riscv_vand_vx_i32m1(quadrant, 2, vl), 0, vl);
  // sign[] is {1,-1,-1,1}; cosine polynomial sign depends on bit 1.
  auto signbit = __riscv_vxor_vv_i32m1(
      quadrant, __riscv_vsra_vx_i32m1(quadrant, 1, vl), vl);
  auto sine_negative =
      __riscv_vmsne_vx_i32m1_b32(__riscv_vand_vx_i32m1(signbit, 1, vl), 0, vl);
  auto sx = __riscv_vmerge_vvm_f32m1(x, __riscv_vfneg_v_f32m1(x, vl),
                                     sine_negative, vl);
  auto x3 = __riscv_vfmul_vv_f32m1(sx, x2, vl);
  auto s1 =
      __riscv_vfmacc_vf_f32m1(__riscv_vfmv_v_f_f32m1(0x1.1107605230bc4p-7, vl),
                              -0x1.994eb3774cf24p-13, x2, vl);
  auto s = __riscv_vfmacc_vf_f32m1(sx, -0x1.555545995a603p-3, x3, vl);
  auto x7 = __riscv_vfmul_vv_f32m1(x3, x2, vl);
  s = __riscv_vfmacc_vv_f32m1(s, s1, x7, vl);
  auto x4 = __riscv_vfmul_vv_f32m1(x2, x2, vl);
  auto c2 = __riscv_vfmacc_vf_f32m1(
      __riscv_vfmv_v_f_f32m1(-0x1.6c087e89a359dp-10, vl), 0x1.99343027bf8c3p-16,
      x2, vl);
  auto c1 = __riscv_vfmacc_vf_f32m1(__riscv_vfmv_v_f_f32m1(1, vl),
                                    -0x1.ffffffd0c621cp-2, x2, vl);
  auto c = __riscv_vfmacc_vf_f32m1(c1, 0x1.55553e1068f19p-5, x4, vl);
  auto x6 = __riscv_vfmul_vv_f32m1(x4, x2, vl);
  c = __riscv_vfmacc_vv_f32m1(c, c2, x6, vl);
  c = __riscv_vmerge_vvm_f32m1(c, __riscv_vfneg_v_f32m1(c, vl), negative, vl);
  auto odd = __riscv_vmsne_vx_i32m1_b32(__riscv_vand_vx_i32m1(n, 1, vl), 0, vl);
  auto result = __riscv_vmerge_vvm_f32m1(s, c, odd, vl);
  auto subnormal = __riscv_vmsltu_vx_u32m1_b32(top, 0x008, vl);
  if (__riscv_vcpop_m_b32(subnormal, vl)) {
    auto extended = value;
    auto square = __riscv_vfmul_vv_f32m1_m(subnormal, extended, extended, vl);
    auto rounded = square;
    // glibc forces this conversion for its underflow/inexact side effects.
    asm volatile("" : : "vr"(rounded));
  }
  result = __riscv_vmerge_vvm_f32m1(result, value, tiny, vl);
  auto invalid = __riscv_vmnot_m_b32(finite, vl);
  auto difference = __riscv_vfsub_vv_f32m1_m(invalid, value, value, vl);
  auto nan = __riscv_vfdiv_vv_f32m1_m(invalid, difference, difference, vl);
  return __riscv_vmerge_vvm_f32m1(result, nan, invalid, vl);
}
