#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/fp2int.h>
#include <stdint.h>
#include <stdio.h>

static float input[16] __attribute__((aligned(64))) = {
    -10.0f, -5.0f, -2.5f, -1.0f, 0.0f,  1.0f,  2.5f,  5.0f,
    10.0f,  7.5f,  3.0f,  0.5f,  -0.5f, -3.0f, -7.5f, 0.0f,
};
static int8_t output[16] __attribute__((aligned(64)));
static const int8_t expected[16] = {
    -127, -64, -32, -13, 0, 13, 32, 64, 127, 95, 38, 6, -6, -38, -95, 0,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 1, 1);
  bb_fp2int(0, 1, 1, 0);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_fence();
  for (int i = 0; i < 16; ++i) {
    if (output[i] != expected[i]) {
      printf("fp2int_i8_signed i=%d got=%d exp=%d\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("fp2int_i8_signed PASS\n");
  return 0;
}
