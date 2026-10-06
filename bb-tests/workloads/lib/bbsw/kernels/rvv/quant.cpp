#include "kernels.h"
#include <riscv_vector.h>
#include <stdint.h>

// Codes and E8M0 scales occupy separate contiguous regions of the output bank.
extern "C" void rvv_quant(uint8_t *output, const uint32_t *input, size_t count,
                          uint32_t *status) {
  *status = 0;
  for (size_t block = 0; block < count / 32; ++block) {
    uint32_t maximum = 0;
    for (size_t begin = 0; begin < 32;) {
      size_t vl = __riscv_vsetvl_e32m1(32 - begin);
      auto bits = __riscv_vle32_v_u32m1(input + block * 32 + begin, vl);
      auto magnitude = __riscv_vand_vx_u32m1(bits, 0x7fffffff, vl);
      auto seed = __riscv_vmv_v_x_u32m1(maximum, vl);
      maximum = __riscv_vmv_x_s_u32m1_u32(
          __riscv_vredmaxu_vs_u32m1_u32m1(magnitude, seed, vl));
      begin += vl;
    }
    if (maximum >= 0x7f800000) {
      *status = 1;
      return;
    }
    int blockExponent = maximum == 0 ? 0 : int(maximum >> 23) - 135;
    if (blockExponent < -127)
      blockExponent = -127;
    output[count + block] = uint8_t(blockExponent + 127);
    for (size_t begin = 0; begin < 32;) {
      size_t vl = __riscv_vsetvl_e32m1(32 - begin);
      auto bits = __riscv_vle32_v_u32m1(input + block * 32 + begin, vl);
      auto sign =
          __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(bits, 24, vl), 128, vl);
      auto fraction = __riscv_vand_vx_u32m1(bits, 0x7fffff, vl);
      auto exponent = __riscv_vreinterpret_v_u32m1_i32m1(
          __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(bits, 23, vl), 255, vl));
      auto subnormal =
          __riscv_vmand_mm_b32(__riscv_vmseq_vx_i32m1_b32(exponent, 0, vl),
                               __riscv_vmsne_vx_u32m1_b32(fraction, 0, vl), vl);
      if (__riscv_vcpop_m_b32(subnormal, vl)) {
        auto normalized = fraction;
        auto shift = __riscv_vmv_v_x_u32m1(0, vl);
        for (unsigned step = 16; step; step >>= 1) {
          auto selected =
              __riscv_vmsltu_vx_u32m1_b32(normalized, 1U << (24 - step), vl);
          auto moved = __riscv_vsll_vx_u32m1(normalized, step, vl);
          normalized =
              __riscv_vmerge_vvm_u32m1(normalized, moved, selected, vl);
          auto increment = __riscv_vadd_vx_u32m1(shift, step, vl);
          shift = __riscv_vmerge_vvm_u32m1(shift, increment, selected, vl);
        }
        fraction = __riscv_vmerge_vvm_u32m1(
            fraction, __riscv_vand_vx_u32m1(normalized, 0x7fffff, vl),
            subnormal, vl);
        auto adjusted = __riscv_vrsub_vx_i32m1(
            __riscv_vreinterpret_v_u32m1_i32m1(shift), 1, vl);
        exponent = __riscv_vmerge_vvm_i32m1(exponent, adjusted, subnormal, vl);
      }
      exponent = __riscv_vsub_vx_i32m1(exponent, blockExponent + 120, vl);
      auto underflow = __riscv_vmsle_vx_i32m1_b32(exponent, 0, vl);
      vuint32m1_t code;
      if (__riscv_vcpop_m_b32(underflow, vl) == 0) {
        auto parity = __riscv_vand_vx_u32m1(
            __riscv_vsrl_vx_u32m1(fraction, 20, vl), 1, vl);
        auto rounded = __riscv_vsrl_vx_u32m1(
            __riscv_vadd_vv_u32m1(__riscv_vadd_vx_u32m1(fraction, 0x7ffff, vl),
                                  parity, vl),
            20, vl);
        code =
            __riscv_vadd_vv_u32m1(rounded,
                                  __riscv_vreinterpret_v_i32m1_u32m1(
                                      __riscv_vsll_vx_i32m1(exponent, 3, vl)),
                                  vl);
      } else {
        auto shift = __riscv_vmv_v_x_u32m1(20, vl);
        auto underShift = __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vrsub_vx_i32m1(exponent, 21, vl));
        shift = __riscv_vmerge_vvm_u32m1(shift, underShift, underflow, vl);
        auto tooSmall = __riscv_vmsgtu_vx_u32m1_b32(shift, 31, vl);
        shift = __riscv_vminu_vx_u32m1(shift, 31, vl);
        auto value = __riscv_vmerge_vvm_u32m1(
            fraction, __riscv_vor_vx_u32m1(fraction, 0x800000, vl), underflow,
            vl);
        auto half =
            __riscv_vsll_vv_u32m1(__riscv_vmv_v_x_u32m1(1, vl),
                                  __riscv_vsub_vx_u32m1(shift, 1, vl), vl);
        auto parity = __riscv_vand_vx_u32m1(
            __riscv_vsrl_vv_u32m1(value, shift, vl), 1, vl);
        auto rounded = __riscv_vsrl_vv_u32m1(
            __riscv_vadd_vv_u32m1(
                __riscv_vsub_vx_u32m1(__riscv_vadd_vv_u32m1(value, half, vl), 1,
                                      vl),
                parity, vl),
            shift, vl);
        rounded = __riscv_vmerge_vxm_u32m1(rounded, 0, tooSmall, vl);
        auto normal =
            __riscv_vadd_vv_u32m1(rounded,
                                  __riscv_vreinterpret_v_i32m1_u32m1(
                                      __riscv_vsll_vx_i32m1(exponent, 3, vl)),
                                  vl);
        code = __riscv_vmerge_vvm_u32m1(normal, rounded, underflow, vl);
      }
      code = __riscv_vminu_vx_u32m1(code, 126, vl);
      auto zero = __riscv_vmseq_vx_u32m1_b32(
          __riscv_vand_vx_u32m1(bits, 0x7fffffff, vl), 0, vl);
      code = __riscv_vmerge_vxm_u32m1(code, 0, zero, vl);
      code = __riscv_vor_vv_u32m1(code, sign, vl);
      auto narrow = __riscv_vnsrl_wx_u16mf2(code, 0, vl);
      __riscv_vse8_v_u8mf4(output + block * 32 + begin,
                           __riscv_vnsrl_wx_u8mf4(narrow, 0, vl), vl);
      begin += vl;
    }
  }
}
