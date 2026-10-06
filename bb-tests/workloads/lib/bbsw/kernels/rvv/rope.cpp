#include "kernels.h"
#include "math/math.h"

extern "C" void rvv_rope(float *output, const float *input,
                         const float *frequencies, const int64_t *positions,
                         size_t rows, size_t headDimAndHeads) {
  const size_t headDim = headDimAndHeads & 65535, heads = headDimAndHeads >> 16;
  const size_t half = headDim / 2;
  for (size_t row = 0; row < rows; ++row) {
    auto integer = __riscv_vle64_v_i64m2(positions + row, 1);
    auto converted = __riscv_vfncvt_f_x_w_f32m1(integer, 1);
    const float position = __riscv_vfmv_f_s_f32m1_f32(converted);
    for (size_t offset = 0; offset < half;) {
      const size_t vl = __riscv_vsetvl_e32m1(half - offset);
      auto frequency = __riscv_vle32_v_f32m1(frequencies + offset, vl);
      auto angle = __riscv_vfmul_vf_f32m1(frequency, position, vl);
      angle = __riscv_vfadd_vf_f32m1(angle, 0, vl);
      auto cosine = rvv_cos(angle, vl);
      for (size_t head = 0; head < heads; ++head) {
        const float *source = input + (head * rows + row) * headDim;
        float *destination = output + (head * rows + row) * headDim;
        auto lower = __riscv_vle32_v_f32m1(source + offset, vl);
        auto upper = __riscv_vle32_v_f32m1(source + half + offset, vl);
        __riscv_vse32_v_f32m1(destination + offset,
                              __riscv_vfmul_vv_f32m1(lower, cosine, vl), vl);
        __riscv_vse32_v_f32m1(destination + half + offset,
                              __riscv_vfmul_vv_f32m1(upper, cosine, vl), vl);
      }
      frequency = __riscv_vle32_v_f32m1(frequencies + offset, vl);
      angle = __riscv_vfmul_vf_f32m1(frequency, position, vl);
      angle = __riscv_vfadd_vf_f32m1(angle, 0, vl);
      auto sine = rvv_sin(angle, vl);
      for (size_t head = 0; head < heads; ++head) {
        const float *source = input + (head * rows + row) * headDim;
        float *destination = output + (head * rows + row) * headDim;
        auto lower = __riscv_vle32_v_f32m1(source + offset, vl);
        auto upper = __riscv_vle32_v_f32m1(source + half + offset, vl);
        auto rotatedLower =
            __riscv_vfmul_vv_f32m1(__riscv_vfneg_v_f32m1(upper, vl), sine, vl);
        auto rotatedUpper = __riscv_vfmul_vv_f32m1(lower, sine, vl);
        auto lowerProduct = __riscv_vle32_v_f32m1(destination + offset, vl);
        auto upperProduct =
            __riscv_vle32_v_f32m1(destination + half + offset, vl);
        __riscv_vse32_v_f32m1(
            destination + offset,
            __riscv_vfadd_vv_f32m1(lowerProduct, rotatedLower, vl), vl);
        __riscv_vse32_v_f32m1(
            destination + half + offset,
            __riscv_vfadd_vv_f32m1(upperProduct, rotatedUpper, vl), vl);
      }
      offset += vl;
    }
  }
}

extern "C" void rvv_rope_launch(float *output, const float *input,
                                const float *frequencies,
                                const int64_t *positions, size_t rows,
                                size_t headDimAndHeads, uint32_t *flags,
                                uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  rvv_rope(output, input, frequencies, positions, rows, headDimAndHeads);
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
