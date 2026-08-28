#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[4] __attribute__((aligned(64))) = {8, -4, 16, 0};
static uint32_t output[4] __attribute__((aligned(64))) = {
    0xdeadbeef,
    0xdeadbeef,
    0xdeadbeef,
    0xdeadbeef,
};
static float da[4] __attribute__((aligned(64))) = {0.5f};
static float dw[4] __attribute__((aligned(64))) = {0.25f};
static const uint32_t expected[4] = {
    0x3f800000,
    0xbf000000,
    0x40000000,
    0x00000000,
};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  bb_mvin_mmio((uintptr_t)da, 0, 1, 4);
  bb_mvin_mmio((uintptr_t)dw, 16, 1, 4);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 1, 1);
  bb_int2fp_tensor(0, 1, 1, 0, 16);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_fence();
  for (int i = 0; i < 4; ++i) {
    if (output[i] != expected[i]) {
      printf("int2fp_tensor_values i=%d got=%08x exp=%08x\n", i, output[i],
             expected[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("int2fp_tensor_values PASS\n");
  return 0;
}
