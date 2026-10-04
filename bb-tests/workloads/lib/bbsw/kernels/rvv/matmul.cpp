#include "kernels.h"
#include <riscv_vector.h>

extern "C" void rvv_matmul(float *output, const float *lhs, const float *rhs,
                           size_t rows, size_t cols, size_t inner) {
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols;) {
      const size_t vl = __riscv_vsetvl_e32m1(cols - col);
      auto sum = __riscv_vle32_v_f32m1(output + row * cols + col, vl);
      for (size_t k = 0; k < inner; ++k) {
        auto weight = __riscv_vle32_v_f32m1(rhs + k * cols + col, vl);
        auto product = __riscv_vfmul_vf_f32m1(weight, lhs[row * inner + k], vl);
        sum = __riscv_vfadd_vv_f32m1(sum, product, vl);
      }
      __riscv_vse32_v_f32m1(output + row * cols + col, sum, vl);
      col += vl;
    }
  }
}
