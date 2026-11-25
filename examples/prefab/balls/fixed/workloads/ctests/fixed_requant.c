#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
static int32_t input[32] __attribute__((aligned(64)));
static int32_t parameters[32] __attribute__((aligned(64)));
static int32_t output[32] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 32; ++i)
    input[i] = i - 17;
  input[0] = INT32_MIN;
  input[1] = INT32_MAX;
  for (int i = 0; i < 16; ++i) {
    parameters[i] = i < 2 ? 1 << 30 : i == 7 ? 0 : 1;
    parameters[16 + i] = i < 2 ? 0 : 1;
  }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 8, 1);
  bb_mvin((uintptr_t)parameters, 1, 8, 1);
  bb_fixed(0, 1, 2, 8, BB_FIXED_REQUANT, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 8, 1);
  for (int i = 0; i < 32; ++i) {
    int64_t product = (int64_t)input[i] * parameters[i % 16];
    uint64_t magnitude = product < 0 ? -product : product;
    uint64_t divisor = UINT64_C(1) << parameters[16 + i % 16];
    uint64_t q = magnitude / divisor, r = magnitude % divisor;
    q += r * 2 > divisor || (r * 2 == divisor && (q & 1));
    int64_t expected = product < 0 ? -(int64_t)q : (int64_t)q;
    if (expected > INT32_MAX)
      expected = INT32_MAX;
    if (expected < INT32_MIN)
      expected = INT32_MIN;
    if (output[i] != expected)
      return 1;
  }
  printf("Fixed requant PASSED\n");
  return 0;
}
