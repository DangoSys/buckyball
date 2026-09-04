#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[32] __attribute__((aligned(64))) = {
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
};
static float output[32] __attribute__((aligned(64)));
static const float scale[16] __attribute__((aligned(64))) = {
    0.125f,  0.25f,  0.375f,  0.5f,  0.625f, 0.75f, 0.875f, 1.0f,
    0.0625f, 0.125f, 0.1875f, 0.25f, 1.125f, 1.25f, 1.375f, 1.5f,
};
static const float expected[32] = {
    1, 2, 3, 4, 5,  6,  7,  8,  0.5f, 1, 1.5f, 2, 9,  10, 11, 12,
    2, 4, 6, 8, 10, 12, 14, 16, 1,    2, 3,    4, 18, 20, 22, 24,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mem_alloc(0, 8, 1);
  bb_mem_alloc(1, 4, 1);
  bb_mem_alloc(2, 8, 1);
  bb_mvin((uintptr_t)input, 0, 8, 1);
  bb_mvin((uintptr_t)scale, 1, 4, 1);
  bb_int32_to_fp32(0, 1, 2, 8, 0);
  bb_mvout((uintptr_t)output, 2, 8, 1);
  bb_fence();
  for (int i = 0; i < 32; ++i) {
    if (output[i] != expected[i]) {
      printf("int2fp_channel_two_rows i=%d got=%f exp=%f\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("int2fp_channel_two_rows PASS\n");
  return 0;
}
