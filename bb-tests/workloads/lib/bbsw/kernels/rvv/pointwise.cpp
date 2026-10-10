#include "kernels.h"
#include <riscv_vector.h>

extern "C" void rvv_pointwise(float *output, const float *lhs, const float *rhs,
                              size_t count, uint32_t operation, uint32_t *flags,
                              uint32_t rounding) {
  asm volatile("csrw fcsr, %0" ::"r"(rounding) : "memory");
  for (size_t first = 0; first < count;) {
    size_t vl = __riscv_vsetvl_e32m1(count - first);
    auto a = __riscv_vle32_v_f32m1(lhs + first, vl);
    vfloat32m1_t result;
    if (operation == 4) {
      result = __riscv_vfneg_v_f32m1(a, vl);
    } else if (operation == 5) {
      result = __riscv_vfabs_v_f32m1(a, vl);
    } else {
      auto b = __riscv_vle32_v_f32m1(rhs + first, vl);
      switch (operation) {
      case 0:
        result = __riscv_vfadd_vv_f32m1(a, b, vl);
        break;
      case 1:
        result = __riscv_vfsub_vv_f32m1(a, b, vl);
        break;
      case 2:
        result = __riscv_vfmul_vv_f32m1(a, b, vl);
        break;
      case 3:
        result = __riscv_vfdiv_vv_f32m1(a, b, vl);
        break;
      case 6:
      case 7: {
        result = operation == 6 ? __riscv_vfmax_vv_f32m1(a, b, vl)
                                : __riscv_vfmin_vv_f32m1(a, b, vl);
        auto nan =
            __riscv_vmor_mm_b32(__riscv_vmfne_vv_f32m1_b32(a, a, vl),
                                __riscv_vmfne_vv_f32m1_b32(b, b, vl), vl);
        auto quiet = __riscv_vfadd_vv_f32m1_m(nan, a, b, vl);
        result = __riscv_vmerge_vvm_f32m1(result, quiet, nan, vl);
        break;
      }
      default:
        __builtin_trap();
      }
    }
    __riscv_vse32_v_f32m1(output + first, result, vl);
    first += vl;
  }
  uint32_t status;
  asm volatile("csrr %0, fflags" : "=r"(status)::"memory");
  *flags = status;
}
