#include "kernels.h"

extern "C" float rvv_erff(float value);

extern "C" void rvv_gelu(float *output, const float *input, size_t count,
                         uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t index = 0; index < count; ++index) {
    const float value = input[index];
    const float scaled = value * 0x1.6a09e6p-1f;
    const float error = rvv_erff(scaled);
    const float shifted = error + 1.0f;
    const float halved = value * 0.5f;
    output[index] = halved * shifted;
  }
  uint32_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = result;
}
