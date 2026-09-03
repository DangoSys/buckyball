#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/toint8.h>
#include <stdint.h>
#include <stdio.h>

static float input[32] __attribute__((aligned(64))) = {
    -8.0f, -7.0f,  -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f,  127.0f, 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
    7.0f,  8.0f,   -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f,
    -2.0f, -1.0f,  0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
};
static int8_t output[32] __attribute__((aligned(64)));
static const int8_t expected[32] = {
    -8, -7, -6, -5, -4, -3, -2, -1, 0,  127, 1, 2, 3, 4, 5, 6,
    7,  8,  -8, -7, -6, -5, -4, -3, -2, -1,  0, 1, 2, 3, 4, 5,
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
      printf("toint8_i8_rows i=%d got=%d exp=%d\n", i, output[i], expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("toint8_i8_rows PASS\n");
  return 0;
}
