#include "buckyball.h"
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { BYTES = BANK_LINES * BANK_WIDTH / 8 };
static uint8_t a[2][BYTES] __attribute__((aligned(64)));
static uint8_t packed[BYTES] __attribute__((aligned(64)));
static uint8_t w[BYTES] __attribute__((aligned(64)));
static uint8_t actual[BYTES] __attribute__((aligned(64)));
static uint8_t expected[BYTES] __attribute__((aligned(64)));
int main(void) {
  memset(a, 255, sizeof(a));
  memset(a[0], 0x38, 32);
  a[0][64] = 127;
  memset(a[1] + 32, 0x40, 32);
  a[1][97] = 127;
  for (int col = 0; col < 16; ++col) {
    memset(w + col * 32, col & 1 ? 0xb8 : 0x38, 32);
    w[512 + col] = 127;
  }
  memset(actual, 0x5a, BYTES);
  memset(expected, 0x5a, BYTES);
  for (int bank = 3; bank <= 8; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a[0], 3, BANK_LINES, 1);
  bb_mvin((uintptr_t)a[1], 4, BANK_LINES, 1);
  bb_mvin((uintptr_t)w, 6, BANK_LINES, 1);
  bb_mvin((uintptr_t)expected, 7, BANK_LINES, 1);
  bb_mvin((uintptr_t)actual, 8, BANK_LINES, 1);
  for (int mode = 0; mode < 2; ++mode)
    for (int part = 0; part < 2; ++part) {
      if (mode == 0) {
        memcpy(packed, a[part] + part * 32, 32);
        packed[32] = 127;
        bb_mvin((uintptr_t)packed, 5, 3, 1);
        bb_mxmm_mxfp8(5, 6, 7, 1, 16, 32, part == 0, part == 1, 1);
        bb_mvout((uintptr_t)expected, 7, BANK_LINES, 1);
      } else {
        bb_mxmm_mxfp8_window(3 + part, 6, 8, 1, 16, 32, part == 0, part == 1, 1,
                             part ? 96 : 64, part * 32);
        bb_mvout((uintptr_t)actual, 8, BANK_LINES, 1);
      }
      if (part == 0)
        for (int i = 0; i < BYTES; ++i)
          if ((mode ? actual : expected)[i] != 0x5a)
            return 1;
    }
  for (int col = 0; col < 16; ++col) {
    uint32_t bits;
    memcpy(&bits, expected + 16 + col * 4, 4);
    if (bits != (col & 1 ? 0xc2c00000u : 0x42c00000u))
      return 4;
  }
  if (memcmp(actual, expected, BYTES))
    return 2;
  for (int i = 0; i < BYTES; ++i)
    if ((i < 16 || i >= 80) && actual[i] != 0x5a)
      return 3;
  for (int bank = 3; bank <= 8; ++bank)
    bb_mem_release(bank);
  puts("mxmm A-window continuation bank/layout/poison/guard PASS");
  return 0;
}
