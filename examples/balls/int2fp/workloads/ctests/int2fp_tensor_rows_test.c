#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[8]
    __attribute__((aligned(64))) = {4, -4, 8, -8, 12, -12, 16, -16};
static float output[8] __attribute__((aligned(64)));
static float da[4] __attribute__((aligned(64))) = {0.5f};
static float dw[4] __attribute__((aligned(64))) = {0.5f};
static const float expected[8] = {1.0f, -1.0f, 2.0f, -2.0f,
                                  3.0f, -3.0f, 4.0f, -4.0f};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mvin_mmio((uintptr_t)da, 0, 1, 4);
  bb_mvin_mmio((uintptr_t)dw, 16, 1, 4);
  bb_mem_alloc(0, 2, 1);
  bb_mem_alloc(1, 2, 1);
  bb_mvin((uintptr_t)input, 0, 2, 1);
  bb_int2fp_tensor(0, 1, 2, 0, 16);
  bb_mvout((uintptr_t)output, 1, 2, 1);
  bb_fence();
  for (int i = 0; i < 8; ++i) {
    if (output[i] != expected[i]) {
      printf("int2fp_tensor_rows i=%d got=%f exp=%f\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("int2fp_tensor_rows PASS\n");
  return 0;
}
