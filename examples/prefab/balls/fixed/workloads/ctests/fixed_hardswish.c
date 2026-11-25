#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
enum { Q = 1 << 24 };
static const int32_t input[16] __attribute__((aligned(64))) = {
    0,     Q / 2,  Q,     -Q / 2, -Q,        -3 * Q / 2, 2 * Q,     -2 * Q,
    3 * Q, -3 * Q, 4 * Q, -4 * Q, INT32_MAX, INT32_MIN,  3 * Q - 1, -3 * Q + 1};
static int32_t parameters[32] __attribute__((aligned(64)));
static int32_t output[16] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 16; ++i)
    parameters[i] = 1;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_mvin((uintptr_t)parameters, 1, 8, 1);
  bb_fixed(0, 1, 2, 4, BB_FIXED_HSWISH, 24, 0, 0);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  for (int i = 0; i < 16; ++i) {
    int64_t x = input[i], expected;
    if (x <= -3 * Q)
      expected = 0;
    else if (x >= 3 * Q)
      expected = x;
    else {
      int64_t product = x * (x + 3 * Q);
      uint64_t n = product < 0 ? -product : product;
      uint64_t q = n / (6 * Q), r = n % (6 * Q);
      q += r * 2 > 6 * Q || (r * 2 == 6 * Q && (q & 1));
      expected = product < 0 ? -(int64_t)q : (int64_t)q;
    }
    if (output[i] != expected)
      return 1;
  }
  printf("Fixed hardswish PASSED\n");
  return 0;
}
