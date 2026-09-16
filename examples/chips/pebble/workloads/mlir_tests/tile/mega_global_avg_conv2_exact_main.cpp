#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <params.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

enum {
  PHYSICAL_BANKS = BANK_NUM,
  BANK_ROWS = BANK_LINES,
  ROW_BYTES = BANK_WIDTH / 8
};
static_assert(
    PHYSICAL_BANKS > 5 && VIRTUAL_BANK_NUM >= PHYSICAL_BANKS,
    "bank reuse regression requires physical bank 5 to be addressable");
static elem_t stale_bank[BANK_ROWS * ROW_BYTES] __attribute__((aligned(64)));

extern "C" void poison_reused_bank(void) {
  for (size_t i = 0; i < sizeof(stale_bank); ++i)
    stale_bank[i] = 0x5a;
  for (int bank = 0; bank < PHYSICAL_BANKS; ++bank)
    bb_mem_alloc(bank, 1, 1);
  for (int bank = 5; bank < 6; ++bank)
    bb_mvin((uintptr_t)stale_bank, bank, BANK_ROWS, 1);
  bb_fence();
  for (int bank = 0; bank < PHYSICAL_BANKS; ++bank)
    bb_mem_release(bank);
  bb_fence();
}

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t n, int64_t height, int64_t width,
                             int64_t channels, int64_t n_stride,
                             int64_t height_stride, int64_t width_stride,
                             int64_t channel_stride) {
  (void)allocated;
  if (n != 1 || height != 14 || width != 14 || channels != 40 ||
      n_stride != 7840 || height_stride != 560 || width_stride != 40 ||
      channel_stride != 1)
    fail();
  const int8_t *output = aligned + offset;
  uint64_t hash = 1469598103934665603ULL;
  for (int y = 0; y < 14; ++y)
    for (int x = 0; x < 14; ++x)
      for (int c = 0; c < 40; ++c) {
        hash ^= (uint8_t)output[y * height_stride + x * width_stride + c];
        hash *= 1099511628211ULL;
      }
  if (hash != 0xad0f766db5880972ULL) {
    printf("FAILED: mega_global_avg_conv2_exact hash=%08x%08x\n",
           (uint32_t)(hash >> 32), (uint32_t)hash);
    fail();
  }
  printf("PASSED: mega_global_avg_conv2_exact\n");
}
