#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

extern "C" void gather(uint32_t *output, const uint32_t *input,
                       const uint32_t *indices, size_t mode, size_t index,
                       uint64_t *length) {
  const size_t vl = __riscv_vsetvl_e32m1(32);
  auto values = __riscv_vle32_v_u32m1(input, vl);
  auto order = __riscv_vle32_v_u32m1(indices, vl);
  vuint32m1_t result;
  switch (mode) {
  case 0:
    result = __riscv_vrgather_vx_u32m1(values, 0, vl);
    break;
  case 1:
    result = __riscv_vrgather_vx_u32m1(values, 16, vl);
    break;
  case 2:
    result = __riscv_vrgather_vx_u32m1(values, 31, vl);
    break;
  case 3:
    result = __riscv_vrgather_vx_u32m1(values, index, vl);
    break;
  case 4:
    result = __riscv_vrgather_vv_u32m1(values, order, vl);
    break;
  case 5: {
    auto mask = __riscv_vmseq_vx_u32m1_b32(order, 0, vl);
    auto previous = __riscv_vmv_v_x_u32m1(UINT32_C(0xdeadbeef), vl);
    result = __riscv_vrgather_vx_u32m1_mu(mask, previous, values, 0, vl);
    break;
  }
  default:
    __builtin_trap();
  }
  __riscv_vse32_v_u32m1(output, result, vl);
  *length = vl;
}
