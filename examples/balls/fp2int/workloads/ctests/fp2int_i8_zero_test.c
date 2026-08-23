#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/fp2int.h>
#include <stdint.h>
#include <stdio.h>

static float input[16] __attribute__((aligned(64)));
static int8_t output[16] __attribute__((aligned(64)));

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
    if (output[i] != 0) {
      printf("fp2int_i8_zero i=%d got=%d\n", i, output[i]);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("fp2int_i8_zero PASS\n");
  return 0;
}
