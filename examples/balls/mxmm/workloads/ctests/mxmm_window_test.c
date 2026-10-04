#include "buckyball.h"
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint8_t a[80] __attribute__((aligned(64)));
static uint8_t packed[48] __attribute__((aligned(64)));
static uint8_t w[528] __attribute__((aligned(64)));
static uint8_t actual[96] __attribute__((aligned(64)));
static uint8_t expected[96] __attribute__((aligned(64)));

int main(void) {
  const uint8_t codes[] = {0,    0x80, 1,    0x81, 0x38,
                           0xb8, 0x37, 0xb7, 0x7e, 0xfe};
  memset(a, 255, sizeof(a));
  for (int i = 0; i < 32; ++i)
    a[32 + i] = codes[i % 10];
  a[65] = 127;
  memcpy(packed, a + 32, 32);
  packed[32] = a[65];
  for (int col = 0; col < 16; ++col) {
    for (int i = 0; i < 32; ++i)
      w[col * 32 + i] = codes[(i + col + 4) % 10];
    w[512 + col] = 127;
  }
  memset(actual, 0x5a, sizeof(actual));
  memset(expected, 0x5a, sizeof(expected));
  bb_dma_fence();
  for (int bank = 3; bank <= 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a, 3, sizeof(a) / 16, 1);
  bb_mvin((uintptr_t)packed, 4, sizeof(packed) / 16, 1);
  bb_mvin((uintptr_t)w, 6, sizeof(w) / 16, 1);
  bb_mvin((uintptr_t)expected, 5, sizeof(expected) / 16, 1);
  bb_mvin((uintptr_t)actual, 7, sizeof(actual) / 16, 1);
  bb_mxmm_mxfp8(4, 6, 5, 1, 16, 32, 1, 1, 1);
  bb_mxmm_mxfp8_window(3, 6, 7, 1, 16, 32, 1, 1, 1, 64, 32);
  bb_mvout((uintptr_t)expected, 5, sizeof(expected) / 16, 1);
  bb_mvout((uintptr_t)actual, 7, sizeof(actual) / 16, 1);
  bb_fence();
  if (memcmp(actual, expected, sizeof(actual)))
    return 1;
  for (int i = 0; i < 96; ++i)
    if ((i < 16 || i >= 80) && actual[i] != 0x5a)
      return 2;
  for (int bank = 3; bank <= 7; ++bank)
    bb_mem_release(bank);
  puts("mxmm A-window small old71/75/poison/guard PASS");
  return 0;
}
