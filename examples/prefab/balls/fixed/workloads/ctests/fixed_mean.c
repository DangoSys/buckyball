#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
// Exact division with positive/negative ties, wide sums and I32 saturation.
static int64_t sums[16]
    __attribute__((aligned(64))) = {1,
                                    3,
                                    -1,
                                    -3,
                                    5,
                                    7,
                                    -5,
                                    -7,
                                    (int64_t)INT32_MAX * 3136,
                                    (int64_t)INT32_MIN * 3136,
                                    100,
                                    -100,
                                    INT64_MAX,
                                    INT64_MIN,
                                    0,
                                    -49};
static int32_t divisors[16] __attribute__((aligned(64))) = {
    2, 2, 2, 2, 2, 2, 2, 2, 3136, 3136, 49, 49, 1, 1, 196, 49};
static const int32_t expected[16] = {
    0,         2,         0, -2, 2,         4,         -2, -4,
    INT32_MAX, INT32_MIN, 2, -2, INT32_MAX, INT32_MIN, 0,  -1};
static int32_t output[16] __attribute__((aligned(64)));
int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)sums, 0, 8, 1);
  bb_mvin((uintptr_t)divisors, 1, 4, 1);
  bb_fixed(0, 1, 2, 4, BB_FIXED_MEAN, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  for (int c = 0; c < 16; ++c)
    if (output[c] != expected[c])
      return 1;
  printf("Fixed mean PASSED\n");
  return 0;
}
