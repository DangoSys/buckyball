#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/fixed.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
enum { Q = 1 << 24 };
static int32_t input[32] __attribute__((aligned(64)));
static int32_t gate[16] __attribute__((aligned(64)));
static int32_t output[32] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 32; ++i)
    input[i] = (i - 16) * (Q / 8) + 1;
  for (int i = 0; i < 16; ++i)
    gate[i] = (i - 8) * (Q / 8);
  input[0] = INT32_MIN;
  gate[0] = INT32_MIN;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 8, 1);
  bb_mvin((uintptr_t)gate, 1, 4, 1);
  bb_fixed(0, 1, 2, 8, BB_FIXED_MUL, 24, 1, 0);
  bb_mvout((uintptr_t)output, 2, 8, 1);
  for (int i = 0; i < 32; ++i) {
    int64_t p = (int64_t)input[i] * gate[i % 16];
    uint64_t n = p < 0 ? -p : p, q = n / Q, r = n % Q;
    q += r * 2 > Q || (r * 2 == Q && (q & 1));
    int64_t expected = p < 0 ? -(int64_t)q : (int64_t)q;
    if (expected > INT32_MAX)
      expected = INT32_MAX;
    if (expected < INT32_MIN)
      expected = INT32_MIN;
    if (output[i] != expected)
      return 1;
  }
  printf("Fixed broadcast multiply PASSED\n");
  return 0;
}
