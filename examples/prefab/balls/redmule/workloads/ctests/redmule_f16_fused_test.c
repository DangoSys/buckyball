#include <isa/redmule.h>
#include <stdio.h>

static const uint16_t a[8]
    __attribute__((aligned(64))) = {0x3c00, 0x3c01, 0, 0, 0, 0, 0, 0};
static const uint16_t b[64] __attribute__((aligned(64))) = {
    0xbc00, 0xbc00, 0xbc00, 0xbc00, 0xbc00, 0xbc00, 0xbc00, 0xbc00,
    0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe};
static uint16_t c[8] __attribute__((aligned(64)));

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)a, 0, 1, 1);
  bb_mvin((uintptr_t)b, 1, 8, 1);
  bb_redmule(0, 1, 2, 1, 8, 0, 0, 0, 1);
  bb_mvout((uintptr_t)c, 2, 1, 1);
  // (1 + 2^-10) * (1 - 2^-10) - 1 = -2^-20; separate multiply/add loses it.
  for (unsigned i = 0; i < 8; ++i)
    if (c[i] != 0x8010)
      __builtin_trap();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  puts("RedMulE FP16 fused cancellation PASS");
  return 0;
}
