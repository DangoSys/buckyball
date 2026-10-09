#include "math/math.h"
#include <stdint.h>

template <bool sine>
static void trig(float *output, const float *input, size_t count,
                 uint32_t *flags, uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t offset = 0; offset < count;) {
    size_t vl = __riscv_vsetvl_e32m1(count - offset);
    auto values = __riscv_vle32_v_f32m1(input + offset, vl);
    auto result = sine ? rvv_sin(values, vl) : rvv_cos(values, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += vl;
  }
  uint32_t status;
  asm volatile("csrr %0, fflags" : "=r"(status)::"memory");
  *flags = status;
}
extern "C" void rvv_sin_launch(float *output, const float *input, size_t count,
                               uint32_t *flags, uint32_t rounding) {
  trig<true>(output, input, count, flags, rounding);
}
extern "C" void rvv_cos_launch(float *output, const float *input, size_t count,
                               uint32_t *flags, uint32_t rounding) {
  trig<false>(output, input, count, flags, rounding);
}
