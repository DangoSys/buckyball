#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[16] __attribute__((aligned(64))) = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
static float output[16] __attribute__((aligned(64)));
static float da[4] __attribute__((aligned(64))) = {0.5f};
static float dw[16] __attribute__((aligned(64))) = {
    0.25f,  0.5f,  0.75f,  1.0f, 1.25f, 1.5f, 1.75f, 2.0f,
    0.125f, 0.25f, 0.375f, 0.5f, 2.25f, 2.5f, 2.75f, 3.0f,
};
static const float expected[16] = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  7.0f,  8.0f,
    0.5f, 1.0f, 1.5f, 2.0f, 9.0f, 10.0f, 11.0f, 12.0f,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mvin_mmio((uintptr_t)da, 0, 1, 4);
  bb_mvin_mmio((uintptr_t)dw, 16, 4, 16);
  bb_mem_alloc(0, 4, 1);
  bb_mem_alloc(1, 4, 1);
  bb_mem_alloc(2, 4, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_int32_to_fp32(0, 1, 2, 4, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  bb_fence();
  for (int i = 0; i < 16; ++i) {
    if (output[i] != expected[i]) {
      printf("int2fp_channel_lanes i=%d got=%f exp=%f\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("int2fp_channel_lanes PASS\n");
  return 0;
}
