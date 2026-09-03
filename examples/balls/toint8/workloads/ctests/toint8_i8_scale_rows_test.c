#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/toint8.h>
#include <stdint.h>
#include <stdio.h>

static float input[32] __attribute__((aligned(64))) = {
    1,  -1,  2,  -2,  3,  -3,  4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
    20, -20, 10, -10, 15, -15, 5, -5, 1, -1, 2, -2, 3, -3, 4, -4,
};
static int8_t output[32] __attribute__((aligned(64)));
static const int8_t expected[32] = {
    6,   -6,   13, -13, 19, -19, 25, -25, 32, -32, 38, -38, 44, -44, 51, -51,
    127, -127, 64, -64, 95, -95, 32, -32, 6,  -6,  13, -13, 19, -19, 25, -25,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mem_alloc(0, 2, 4);
  bb_mem_alloc(1, 2, 1);
  bb_mvin((uintptr_t)input, 0, 2, 1);
  bb_toint8(0, 1, 2, 0);
  bb_mvout((uintptr_t)output, 1, 2, 1);
  bb_fence();
  for (int i = 0; i < 32; ++i) {
    if (output[i] != expected[i]) {
      printf("toint8_i8_scale_rows i=%d got=%d exp=%d\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("toint8_i8_scale_rows PASS\n");
  return 0;
}
