#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/fp2int.h>
#include <stdint.h>
#include <stdio.h>

static float input[16] __attribute__((aligned(64))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static int8_t output[16] __attribute__((aligned(64)));
static const int8_t expected[16] = {
    8, 16, 24, 32, 40, 48, 56, 64, 71, 79, 87, 95, 103, 111, 119, 127,
};

int main(void) {
  bb_mem_alloc(0, 4, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_fp2int(0, 1, 4, 0);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_fence();
  for (int i = 0; i < 16; ++i) {
    if (output[i] != expected[i]) {
      printf("fp2int_i8_stream_1x4 i=%d got=%d exp=%d\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("fp2int_i8_stream_1x4 PASS\n");
  return 0;
}
