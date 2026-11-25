#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
// Two pixels, sixteen channels, then a second tile with the previous I64 sum.
static int32_t input[32] __attribute__((aligned(64)));
static int64_t previous[16] __attribute__((aligned(64)));
static int64_t output[16] __attribute__((aligned(64)));
int main(void) {
  for (int c = 0; c < 16; ++c) {
    input[c] = c % 2 ? INT32_MIN : INT32_MAX;
    input[16 + c] = c - 8;
    previous[c] = (int64_t)(c - 8) * 4294967296LL;
  }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 8, 1);
  bb_mvin((uintptr_t)previous, 1, 8, 1);
  bb_fixed(0, 1, 2, 8, BB_FIXED_SUM, 24, 0, 0);
  bb_fixed(0, 2, 2, 8, BB_FIXED_SUM, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 8, 1);
  for (int c = 0; c < 16; ++c)
    if (output[c] != previous[c] + 2 * ((int64_t)input[c] + input[16 + c]))
      return 1;
  printf("Fixed sum PASSED\n");
  return 0;
}
