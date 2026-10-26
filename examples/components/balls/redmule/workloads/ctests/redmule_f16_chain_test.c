#include <isa/redmule.h>
#include <params.h>
#include <stdio.h>

static uint16_t a[16] __attribute__((aligned(64)));
static uint16_t b[64] __attribute__((aligned(64)));
static uint16_t c[16] __attribute__((aligned(64)));

int main(void) {
  unsigned ab = VIRTUAL_BANK_NUM - 6, bb = ab + 1, cb = ab + 2;
  for (unsigned i = 0; i < 16; ++i) {
    a[i] = 0x3c00;
    c[i] = 0x3555;
  }
  for (unsigned i = 0; i < 64; ++i)
    b[i] = 0x3800;
  bb_mem_alloc(ab, 1, 1);
  bb_mem_alloc(bb, 1, 1);
  bb_mem_alloc(cb, 1, 1);
  bb_mvin((uintptr_t)a, ab, 2, 1);
  bb_mvin((uintptr_t)b, bb, 8, 1);
  bb_mvin((uintptr_t)c, cb, 2, 1);
  bb_redmule(ab, bb, cb, 2, 8, 0, 0, 0, 1);
  bb_mvout((uintptr_t)c, cb, 2, 1);
  for (unsigned i = 0; i < 16; ++i)
    if (c[i] != 0x4400)
      __builtin_trap();
  bb_redmule(ab, bb, cb, 2, 8, 0, 0, 0, 0);
  bb_mvout((uintptr_t)c, cb, 2, 1);
  for (unsigned i = 0; i < 16; ++i)
    if (c[i] != 0x4800)
      __builtin_trap();
  bb_redmule(ab, bb, cb, 2, 8, 0, 0, 0, 1);
  bb_mvout((uintptr_t)c, cb, 2, 1);
  for (unsigned i = 0; i < 16; ++i)
    if (c[i] != 0x4400)
      __builtin_trap();
  bb_mem_release(ab);
  bb_mem_release(bb);
  bb_mem_release(cb);
  puts("RedMulE FP16 chain and restart PASS");
  return 0;
}
