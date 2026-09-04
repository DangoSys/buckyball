#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/toint8.h>
#include <stdint.h>
#include <stdio.h>

static float input[32] __attribute__((aligned(64))) = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
};
static int8_t output[32] __attribute__((aligned(64)));
static const int8_t expected[32] = {
    4,  8,  12, 16, 20, 24, 28, 32, 36, 40,  44,  48,  52,  56,  60,  64,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
};

int main(void) {
  bb_mem_alloc(0, 1, 8);
  bb_mem_alloc(1, 2, 1);
  bb_mvin((uintptr_t)input, 0, 1, 1);
  bb_toint8(0, 1, 1, 0);
  bb_mvout((uintptr_t)output, 1, 2, 1);
  bb_fence();
  for (int i = 0; i < 32; ++i) {
    if (output[i] != expected[i]) {
      printf("toint8_i8_stream_8x1 i=%d got=%d exp=%d\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("toint8_i8_stream_8x1 PASS\n");
  return 0;
}
