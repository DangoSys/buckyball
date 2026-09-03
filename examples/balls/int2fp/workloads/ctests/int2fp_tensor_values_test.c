#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[16] __attribute__((aligned(64))) = {
    8, -4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint32_t output[16] __attribute__((aligned(64))) = {
    0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
    0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
    0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
};
static float scale[16] __attribute__((aligned(64))) = {
    0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f,
    0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
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
  bb_mem_alloc(0, 4, 1);
  bb_mem_alloc(1, 4, 1);
  bb_mem_alloc(2, 4, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_mvin((uintptr_t)scale, 1, 4, 1);
  bb_int32_to_fp32(0, 1, 2, 4, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
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
  bb_mem_release(2);
  printf("int2fp_tensor_values PASS\n");
  return 0;
}
