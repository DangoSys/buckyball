#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/toint8.h>
#include <stdint.h>
#include <stdio.h>

static float input[16] __attribute__((aligned(64))) = {
    0.5f, -0.5f, 1.5f, -1.5f, 2.5f,   -2.5f,   3.5f,   -3.5f,
    4.5f, -4.5f, 5.5f, -5.5f, 126.5f, -126.5f, 127.0f, -127.0f,
};
static int8_t output[16] __attribute__((aligned(64)));
static const int8_t expected[16] = {
    0, 0, 2, -2, 2, -2, 4, -4, 4, -4, 6, -6, 126, -126, 127, -127,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 1, 1);
  bb_toint8(0, 1, 1, 0);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_fence();
  for (int i = 0; i < 16; ++i) {
    if (output[i] != expected[i]) {
      printf("toint8_i8_rounding i=%d got=%d exp=%d\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("toint8_i8_rounding PASS\n");
  return 0;
}
