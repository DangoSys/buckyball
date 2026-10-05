#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

extern "C" void rvv_pack(uint32_t *output, const uint32_t *input, size_t fullK,
                         size_t fullN, size_t startK, size_t startN,
                         size_t countK, size_t columns) {
  for (size_t column = 0; column < columns; ++column)
    for (size_t offset = 0; offset < countK;) {
      const size_t vl = __riscv_vsetvl_e32m1(countK - offset);
      auto value = __riscv_vmv_v_x_u32m1(0, vl);
      if (startN + column < fullN)
        value = __riscv_vlse32_v_u32m1(input + (startK + offset) * fullN +
                                           startN + column,
                                       fullN * sizeof(uint32_t), vl);
      __riscv_vse32_v_u32m1(output + column * countK + offset, value, vl);
      offset += vl;
    }
}
