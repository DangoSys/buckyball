#include "flash_attention.h"
#include "math/math.h"

static float exponential(float value) {
  return __riscv_vfmv_f_s_f32m1_f32(
      rvv_exp(__riscv_vfmv_v_f_f32m1(value, 1), 1));
}

extern "C" void
rvv_flash_attention_launch(const float *query, const float *panel,
                           unsigned char *descriptor, float *output,
                           float *blockOutput, uint64_t dimensions,
                           uint64_t parameters, uint32_t *flags) {
  const unsigned phase = dimensions & 255, rows = (dimensions >> 8) & 255;
  const unsigned width = (dimensions >> 16) & 65535;
  const unsigned begin = (dimensions >> 32) & 255;
  const unsigned count = (dimensions >> 40) & 255;
  const unsigned keyCount = (dimensions >> 48) & 255;
  const uint32_t rounding = parameters >> 32;
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  union {
    uint32_t bits;
    float value;
  } scale{uint32_t(parameters)};
  auto *state =
      reinterpret_cast<rvv_flash::State *>(descriptor + rvv_flash::stateOffset);
  auto *mask = reinterpret_cast<float *>(descriptor + rvv_flash::maskOffset);
  if (phase == 0) {
    for (unsigned row = 0; row < rows; ++row) {
      state->sum[row] = 0;
      state->maximum[row] = -1.0e30f;
      for (unsigned d = 0; d < width; d += rvv_flash::lanes)
        __riscv_vse32_v_f32m2(output + row * width + d,
                              __riscv_vfmv_v_f_f32m2(0, rvv_flash::lanes),
                              rvv_flash::lanes);
    }
  } else if (phase == 1) {
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned k = 0; k < count; ++k) {
        auto partial = __riscv_vfmv_v_f_f32m2(0, rvv_flash::lanes);
        for (unsigned d = 0; d < width; d += rvv_flash::lanes) {
          auto q =
              __riscv_vle32_v_f32m2(query + row * width + d, rvv_flash::lanes);
          auto key =
              __riscv_vle32_v_f32m2(panel + k * width + d, rvv_flash::lanes);
          partial = __riscv_vfmacc_vv_f32m2(partial, q, key, rvv_flash::lanes);
        }
        auto sum = __riscv_vfredosum_vs_f32m2_f32m1(
            partial, __riscv_vfmv_v_f_f32m1(0, 1), rvv_flash::lanes);
        float score = __riscv_vfmv_f_s_f32m1_f32(sum) * scale.value;
        state->scores[row * rvv_flash::keys + begin + k] =
            score + mask[row * rvv_flash::lanes + k];
      }
  } else if (phase == 2) {
    for (unsigned row = 0; row < rows; ++row) {
      float maximum = -1.0e30f;
      for (unsigned k = 0; k < keyCount; ++k) {
        float score = state->scores[row * rvv_flash::keys + k];
        if (score > maximum)
          maximum = score;
      }
      float sum = 0;
      for (unsigned k = 0; k < keyCount; ++k) {
        float p =
            exponential(state->scores[row * rvv_flash::keys + k] - maximum);
        state->scores[row * rvv_flash::keys + k] = p;
        sum = sum + p;
      }
      state->blockMaximum[row] = maximum;
      state->blockSum[row] = sum;
      float merged =
          maximum > state->maximum[row] ? maximum : state->maximum[row];
      state->mergedMaximum[row] = merged;
      state->alpha[row] = exponential(state->maximum[row] - merged);
      state->beta[row] = exponential(maximum - merged);
    }
  } else if (phase == 3) {
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned d = 0; d < width; d += rvv_flash::lanes) {
        auto partial =
            begin == 0 ? __riscv_vfmv_v_f_f32m2(0, rvv_flash::lanes)
                       : __riscv_vle32_v_f32m2(blockOutput + row * width + d,
                                               rvv_flash::lanes);
        for (unsigned k = 0; k < count; ++k) {
          auto value =
              __riscv_vle32_v_f32m2(panel + k * width + d, rvv_flash::lanes);
          float p = state->scores[row * rvv_flash::keys + begin + k];
          partial =
              __riscv_vfmacc_vf_f32m2(partial, p, value, rvv_flash::lanes);
        }
        __riscv_vse32_v_f32m2(blockOutput + row * width + d, partial,
                              rvv_flash::lanes);
      }
  } else if (phase == 4) {
    for (unsigned row = 0; row < rows; ++row) {
      for (unsigned d = 0; d < width; d += rvv_flash::lanes) {
        auto old =
            __riscv_vle32_v_f32m2(output + row * width + d, rvv_flash::lanes);
        auto block = __riscv_vle32_v_f32m2(blockOutput + row * width + d,
                                           rvv_flash::lanes);
        old = __riscv_vfmul_vf_f32m2(old, state->alpha[row], rvv_flash::lanes);
        block =
            __riscv_vfmul_vf_f32m2(block, state->beta[row], rvv_flash::lanes);
        auto merged = __riscv_vfadd_vv_f32m2(old, block, rvv_flash::lanes);
        __riscv_vse32_v_f32m2(output + row * width + d, merged,
                              rvv_flash::lanes);
      }
      float old = state->sum[row] * state->alpha[row];
      float block = state->blockSum[row] * state->beta[row];
      state->sum[row] = old + block;
      state->maximum[row] = state->mergedMaximum[row];
    }
  } else if (phase == 5) {
    for (unsigned row = 0; row < rows; ++row) {
      for (unsigned d = 0; d < width; d += rvv_flash::lanes) {
        auto value =
            __riscv_vle32_v_f32m2(output + row * width + d, rvv_flash::lanes);
        value =
            __riscv_vfdiv_vf_f32m2(value, state->sum[row], rvv_flash::lanes);
        __riscv_vse32_v_f32m2(output + row * width + d, value,
                              rvv_flash::lanes);
      }
      mask[row] = state->sum[row];
    }
  } else {
    asm volatile("ebreak" ::: "memory");
  }
  uint32_t status;
  asm volatile("csrr %0, fflags" : "=r"(status)::"memory");
  *flags = status;
}
