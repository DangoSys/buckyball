#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>

enum { BANK_BYTES = BANK_LINES * BANK_WIDTH / 8, ELEMENTS = BANK_BYTES / 4 };
static float a[ELEMENTS] __attribute__((aligned(BANK_BYTES)));
static float b[ELEMENTS] __attribute__((aligned(BANK_BYTES)));
static uint32_t result[ELEMENTS] __attribute__((aligned(BANK_BYTES)));

int main(void) {
  for (int shape = 0; shape < 2; ++shape)
    for (int fused = 0; fused < 2; ++fused) {
      int m = shape ? 64 : 16;
      int n = m;
      int k = shape ? 64 : 256;
      for (int i = 0; i < ELEMENTS; ++i) {
        a[i] = 1.0f;
        b[i] = 1.0f;
        result[i] = 0x5a5a5a5a;
      }
      for (int bank = 3; bank <= 5; ++bank)
        bb_mem_alloc(bank, 1, 1);
      bb_mvin((uintptr_t)a, 3, BANK_LINES, 1);
      bb_mvin((uintptr_t)b, 4, BANK_LINES, 1);
      bb_mvin((uintptr_t)result, 5, BANK_LINES, 1);
      if (fused)
        bb_mxmm_fma32(3, 4, 5, m, n, k, 1, 1, 0);
      else
        bb_mxmm_f32(3, 4, 5, m, n, k, 1, 1, 0);
      bb_mvout((uintptr_t)result, 5, BANK_LINES, 1);
      union {
        float value;
        uint32_t bits;
      } expected = {.value = (float)k};
      for (int i = 0; i < ELEMENTS; ++i)
        if (result[i] != (i < m * n ? expected.bits : 0x5a5a5a5a))
          return 1;
      for (int bank = 3; bank <= 5; ++bank)
        bb_mem_release(bank);
    }
  puts("mxmm full input/output bank PASS");
  return 0;
}
