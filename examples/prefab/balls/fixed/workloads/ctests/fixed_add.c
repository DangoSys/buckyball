#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
static int32_t a[16] __attribute__((aligned(64)));
static int32_t b[16] __attribute__((aligned(64)));
static int32_t output[16] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 16; ++i) {
    a[i] = i * 17 - 100;
    b[i] = 50 - i * 9;
  }
  a[0] = INT32_MAX;
  b[0] = 1;
  a[1] = INT32_MIN;
  b[1] = -1;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)a, 0, 4, 1);
  bb_mvin((uintptr_t)b, 1, 4, 1);
  bb_fixed(0, 1, 2, 4, BB_FIXED_ADD, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  for (int i = 0; i < 16; ++i) {
    int64_t expected = (int64_t)a[i] + b[i];
    if (expected > INT32_MAX)
      expected = INT32_MAX;
    if (expected < INT32_MIN)
      expected = INT32_MIN;
    if (output[i] != expected)
      return 1;
  }
  printf("Fixed add PASSED\n");
  return 0;
}
