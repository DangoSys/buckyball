#include <isa/redmule.h>
#include <stdio.h>

static uint16_t a[8 * 8] __attribute__((aligned(64)));
static uint16_t b[18 * 8] __attribute__((aligned(64)));
static uint16_t c[7 * 8] __attribute__((aligned(64)));

int main(void) {
  unsigned ab = CORE_VIRTUAL_BANK_COUNT - 3, bb = ab + 1, cb = ab + 2;
  for (unsigned i = 0; i < 8 * 8; ++i)
    a[i] = 0x7e00;
  for (unsigned i = 0; i < 18 * 8; ++i)
    b[i] = 0x7e00;
  for (unsigned i = 0; i < 7 * 8; ++i)
    c[i] = 0x3555;
  for (unsigned r = 0; r < 3; ++r)
    for (unsigned k = 0; k < 16; ++k)
      a[2 * 8 + r * 16 + k] = r == 1 ? 0xc000 : 0x3c00;
  for (unsigned k = 0; k < 16; ++k)
    for (unsigned n = 0; n < 8; ++n)
      b[(2 + k) * 8 + n] = n & 1 ? 0xb800 : 0x3800;
  bb_mem_alloc(ab, 1, 1);
  bb_mem_alloc(bb, 1, 1);
  bb_mem_alloc(cb, 1, 1);
  bb_mvin((uintptr_t)a, ab, 8, 1);
  bb_mvin((uintptr_t)b, bb, 18, 1);
  bb_mvin((uintptr_t)c, cb, 7, 1);
  bb_redmule(ab, bb, cb, 3, 16, 2, 2, 3, 1);
  bb_mvout((uintptr_t)c, cb, 7, 1);
  for (unsigned r = 0; r < 7; ++r)
    for (unsigned n = 0; n < 8; ++n) {
      uint16_t expected = 0x3555;
      if (r >= 3 && r < 6) {
        expected = r == 4 ? 0xcc00 : 0x4800;
        if (n & 1)
          expected ^= 0x8000;
      }
      if (c[r * 8 + n] != expected)
        __builtin_trap();
    }
  bb_mem_release(ab);
  bb_mem_release(bb);
  bb_mem_release(cb);
  puts("RedMulE FP16 nonzero bases and guard rows PASS");
  return 0;
}
