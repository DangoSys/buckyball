#include <isa/redmule.h>
#include <stdio.h>

static const uint16_t a[8]
    __attribute__((aligned(64))) = {0x3c00, 0x1000, 0, 0, 0, 0, 0, 0};
static const uint16_t b[64] __attribute__((aligned(64))) = {
    0x3c00, 0x3c01, 1, 0x7bff, 0x7c00, 0xfc00, 0x7d00, 0x8000, 0x3c00, 0x3c00,
    0,      0x3c00, 0, 0,      0,      0,      0,      0,      0,      0,
    0,      0,      0, 0,      0,      0,      0,      0,      0,      0,
    0,      0,      0, 0,      0,      0,      0,      0,      0,      0,
    0,      0,      0, 0,      0,      0,      0,      0,      0,      0,
    0,      0,      0, 0,      0,      0,      0,      0,      0,      0,
    0,      0,      0, 0};
static const uint16_t expected[8] = {0x3c00, 0x3c02, 1,      0x7bff,
                                     0x7c00, 0xfc00, 0x7e00, 0};
static uint16_t c[8] __attribute__((aligned(64)));

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)a, 0, 1, 1);
  bb_mvin((uintptr_t)b, 1, 8, 1);
  bb_redmule(0, 1, 2, 1, 8, 0, 0, 0, 1);
  bb_mvout((uintptr_t)c, 2, 1, 1);
  for (unsigned i = 0; i < 8; ++i)
    if (c[i] != expected[i])
      __builtin_trap();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  puts("RedMulE FP16 ties, subnormal, infinity and canonical NaN PASS");
  return 0;
}
