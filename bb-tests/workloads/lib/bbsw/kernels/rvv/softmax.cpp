#include "kernels.h"
#include "math/math.h"

extern "C" void rvv_softmax(float *output, const float *input, size_t rows,
                            size_t width) {
  for (size_t row = 0; row < rows; ++row) {
    const float *source = input + row * width;
    float *destination = output + row * width;
    auto maximum = __riscv_vfmv_v_f_f32m1(-0x1.fffffep127f, 1);
    bool hasNaN = false;
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(source + offset, vl);
      auto magnitude = __riscv_vand_vx_u32m1(__riscv_vreinterpret_u32m1(values),
                                             0x7fffffff, vl);
      hasNaN |=
          __riscv_vcpop_m_b32(
              __riscv_vmsgtu_vx_u32m1_b32(magnitude, 0x7f800000, vl), vl) != 0;
      maximum = __riscv_vfredmax_vs_f32m1_f32m1(values, maximum, vl);
      offset += vl;
    }
    if (hasNaN)
      maximum =
          __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0x7fc00000, 1));
    const float peak = __riscv_vfmv_f_s_f32m1_f32(maximum);
    auto sum = __riscv_vfmv_v_f_f32m1(0, 1);
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(source + offset, vl);
      auto centered = __riscv_vfsub_vf_f32m1(values, peak, vl);
      auto exponential = rvv_exp(centered, vl);
      __riscv_vse32_v_f32m1(destination + offset, exponential, vl);
      sum = __riscv_vfredosum_vs_f32m1_f32m1(exponential, sum, vl);
      offset += vl;
    }
    auto reciprocal = __riscv_vfrdiv_vf_f32m1(sum, 1, 1);
    const float scale = __riscv_vfmv_f_s_f32m1_f32(reciprocal);
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(destination + offset, vl);
      auto result = __riscv_vfmul_vf_f32m1(values, scale, vl);
      __riscv_vse32_v_f32m1(destination + offset, result, vl);
      offset += vl;
    }
  }
}

extern "C" void rvv_softmax_launch(float *output, const float *input,
                                   size_t rows, size_t width, uint32_t *flags,
                                   uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_softmax(output, input, rows, width);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}

extern "C" void rvv_attention_softmax_launch(float *output, const float *input,
                                             const uint8_t *mask, size_t rows,
                                             size_t width,
                                             const uint32_t *parameters,
                                             uint32_t *flags,
                                             uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  union {
    uint32_t bits;
    float value;
  } scale{parameters[0]}, masked{parameters[1]};
  for (size_t row = 0; row < rows; ++row)
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(input + row * width + offset, vl);
      auto scaled = __riscv_vfmul_vf_f32m1(values, scale.value, vl);
      auto bytes = __riscv_vle8_v_u8mf4(
          mask + ((row + parameters[3]) % parameters[2]) * width + offset, vl);
      auto selected = __riscv_vmsne_vx_u8mf4_b32(bytes, 0, vl);
      auto result =
          __riscv_vfmerge_vfm_f32m1(scaled, masked.value, selected, vl);
      __riscv_vse32_v_f32m1(output + row * width + offset, result, vl);
      offset += vl;
    }
  rvv_softmax(output, output, rows, width);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}

// Softmax sums are positive and finite for finite scores. Preserve IEEE special
// values when the input row contains infinities or NaNs.
static vfloat32m1_t logSum(vfloat32m1_t value) {
  constexpr size_t vl = 1;
  auto bits = __riscv_vreinterpret_u32m1(value);
  auto fraction = __riscv_vand_vx_u32m1(bits, 0x007fffff, vl);
  auto mantissa = __riscv_vreinterpret_f32m1(
      __riscv_vor_vx_u32m1(fraction, 0x3f800000, vl));
  auto exponent = __riscv_vsub_vx_i32m1(
      __riscv_vreinterpret_i32m1(__riscv_vsrl_vx_u32m1(bits, 23, vl)), 127, vl);
  auto upper = __riscv_vmfge_vf_f32m1_b32(mantissa, 0x1.6a09e6p0f, vl);
  mantissa = __riscv_vmerge_vvm_f32m1(
      mantissa, __riscv_vfmul_vf_f32m1(mantissa, 0.5f, vl), upper, vl);
  exponent = __riscv_vmerge_vvm_i32m1(
      exponent, __riscv_vadd_vx_i32m1(exponent, 1, vl), upper, vl);
  auto numerator = __riscv_vfsub_vf_f32m1(mantissa, 1, vl);
  auto denominator = __riscv_vfadd_vf_f32m1(mantissa, 1, vl);
  auto t = __riscv_vfdiv_vv_f32m1(numerator, denominator, vl);
  auto square = __riscv_vfmul_vv_f32m1(t, t, vl);
  auto term = t, series = t;
  for (unsigned odd = 3; odd <= 15; odd += 2) {
    term = __riscv_vfmul_vv_f32m1(term, square, vl);
    series = __riscv_vfadd_vv_f32m1(
        series, __riscv_vfdiv_vf_f32m1(term, float(odd), vl), vl);
  }
  auto e = __riscv_vfcvt_f_x_v_f32m1(exponent, vl);
  auto result =
      __riscv_vfadd_vv_f32m1(__riscv_vfmul_vf_f32m1(e, 0x1.62e4p-1f, vl),
                             __riscv_vfmul_vf_f32m1(series, 2, vl), vl);
  result = __riscv_vfadd_vv_f32m1(
      result, __riscv_vfmul_vf_f32m1(e, 0x1.7f7d1cp-20f, vl), vl);
  auto zero = __riscv_vmseq_vx_u32m1_b32(bits, 0, vl);
  auto infinity = __riscv_vmseq_vx_u32m1_b32(bits, 0x7f800000, vl);
  auto nan = __riscv_vmsgtu_vx_u32m1_b32(bits, 0x7f800000, vl);
  auto negativeInfinity =
      __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0xff800000, vl));
  auto quietNaN =
      __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0x7fc00000, vl));
  result = __riscv_vmerge_vvm_f32m1(result, negativeInfinity, zero, vl);
  result = __riscv_vmerge_vvm_f32m1(result, value, infinity, vl);
  return __riscv_vmerge_vvm_f32m1(result, quietNaN, nan, vl);
}

extern "C" void rvv_logsumexp_softmax(float *output, const float *input,
                                      size_t rows, size_t width) {
  for (size_t row = 0; row < rows; ++row) {
    const float *source = input + row * width;
    float *destination = output + row * width;
    auto maximum = __riscv_vfmv_v_f_f32m1(-0x1.fffffep127f, 1);
    bool hasNaN = false;
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(source + offset, vl);
      auto magnitude = __riscv_vand_vx_u32m1(__riscv_vreinterpret_u32m1(values),
                                             0x7fffffff, vl);
      hasNaN |=
          __riscv_vcpop_m_b32(
              __riscv_vmsgtu_vx_u32m1_b32(magnitude, 0x7f800000, vl), vl) != 0;
      maximum = __riscv_vfredmax_vs_f32m1_f32m1(values, maximum, vl);
      offset += vl;
    }
    if (hasNaN)
      maximum =
          __riscv_vreinterpret_f32m1(__riscv_vmv_v_x_u32m1(0x7fc00000, 1));
    const float peak = __riscv_vfmv_f_s_f32m1_f32(maximum);
    auto sum = __riscv_vfmv_v_f_f32m1(0, 1);
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(source + offset, vl);
      auto exponential = rvv_exp(__riscv_vfsub_vf_f32m1(values, peak, vl), vl);
      sum = __riscv_vfredosum_vs_f32m1_f32m1(exponential, sum, vl);
      offset += vl;
    }
    auto logsum = __riscv_vfadd_vv_f32m1(maximum, logSum(sum), 1);
    const float normalization = __riscv_vfmv_f_s_f32m1_f32(logsum);
    for (size_t offset = 0; offset < width;) {
      const size_t vl = __riscv_vsetvl_e32m1(width - offset);
      auto values = __riscv_vle32_v_f32m1(source + offset, vl);
      auto result =
          rvv_exp(__riscv_vfsub_vf_f32m1(values, normalization, vl), vl);
      __riscv_vse32_v_f32m1(destination + offset, result, vl);
      offset += vl;
    }
  }
}

extern "C" void rvv_logsumexp_softmax_launch(float *output, const float *input,
                                             size_t rows, size_t width,
                                             uint32_t *flags,
                                             uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_logsumexp_softmax(output, input, rows, width);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
