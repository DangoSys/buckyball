#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <stdint.h>
#include <stdio.h>
static int32_t input[16] __attribute__((aligned(64)));
static int32_t output[16] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 16; ++i)
    input[i] = i - 8;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_fixed(0, 1, 2, 4, BB_FIXED_RELU, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  for (int i = 0; i < 16; ++i)
    if (output[i] != (input[i] > 0 ? input[i] : 0))
      return 1;
  printf("Fixed relu PASSED\n");
  return 0;
}
