#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

extern "C" void strided(uint64_t *output, const uint64_t *input,
                        int64_t input_stride, int64_t output_stride,
                        uint64_t *length) {
  const size_t vl = __riscv_vsetvl_e64m1(16);
  auto values = __riscv_vlse64_v_u64m1(input, input_stride, vl);
  __riscv_vsse64_v_u64m1(output, output_stride, values, vl);
  *length = vl;
}
