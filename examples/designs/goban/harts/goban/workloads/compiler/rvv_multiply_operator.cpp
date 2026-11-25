#include <stddef.h>
#include <stdint.h>

extern "C" void multiply(void *output, const void *left, const void *right,
                         size_t mode, size_t sew, uint64_t seed) {
  const size_t vl = 4;
  asm volatile("vsetvl zero, %0, %1" : : "r"(vl), "r"(sew << 3) : "memory");
  switch (sew) {
  case 0:
    asm volatile("vle8.v v8, (%0)\n vle8.v v10, (%1)"
                 :
                 : "r"(left), "r"(right)
                 : "v8", "v10", "memory");
    break;
  case 1:
    asm volatile("vle16.v v8, (%0)\n vle16.v v10, (%1)"
                 :
                 : "r"(left), "r"(right)
                 : "v8", "v10", "memory");
    break;
  case 2:
    asm volatile("vle32.v v8, (%0)\n vle32.v v10, (%1)"
                 :
                 : "r"(left), "r"(right)
                 : "v8", "v10", "memory");
    break;
  case 3:
    asm volatile("vle64.v v8, (%0)\n vle64.v v10, (%1)"
                 :
                 : "r"(left), "r"(right)
                 : "v8", "v10", "memory");
    break;
  }
  asm volatile("vxor.vx v12, v8, %0" : : "r"(seed) : "v12");
  switch (mode) {
  case 0:
    asm volatile("vmul.vv v12, v8, v10" : : : "v12");
    break;
  case 1:
    asm volatile("vmulhu.vv v12, v10, v8" : : : "v12");
    break;
  case 2:
    asm volatile("vmulhsu.vv v12, v8, v10" : : : "v12");
    break;
  case 3:
    asm volatile("vmulh.vv v12, v10, v8" : : : "v12");
    break;
  case 4:
    asm volatile("vmadd.vv v12, v10, v8" : : : "v12");
    break;
  case 5:
    asm volatile("vnmsub.vv v12, v10, v8" : : : "v12");
    break;
  case 6:
    asm volatile("vmacc.vv v12, v10, v8" : : : "v12");
    break;
  case 7:
    asm volatile("vnmsac.vv v12, v10, v8" : : : "v12");
    break;
  }
  switch (sew) {
  case 0:
    asm volatile("vse8.v v12, (%0)" : : "r"(output) : "memory");
    break;
  case 1:
    asm volatile("vse16.v v12, (%0)" : : "r"(output) : "memory");
    break;
  case 2:
    asm volatile("vse32.v v12, (%0)" : : "r"(output) : "memory");
    break;
  case 3:
    asm volatile("vse64.v v12, (%0)" : : "r"(output) : "memory");
    break;
  }
}
