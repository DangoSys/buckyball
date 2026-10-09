#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>

_Static_assert(BANK_WIDTH == 128 && BANK_LINES >= 64,
               "MVIN2D boundary test requires 128-bit rows and at least 1 KiB");

static const uint8_t input[8 * 256] __attribute__((aligned(128))) = {
    [0 * 256] = 11, [1 * 256] = 22, [2 * 256] = 33, [3 * 256] = 44,
    [4 * 256] = 55, [5 * 256] = 66, [6 * 256] = 77, [7 * 256] = 88,
};
static const uint8_t zeros[64 * BANK_WIDTH / 8] __attribute__((aligned(128)));
static uint8_t output[64 * BANK_WIDTH / 8] __attribute__((aligned(128)));

// MobileNet's 72-byte pixels: the last read straddles the 4 KiB page.
static const uint8_t page_input[8192] __attribute__((aligned(4096))) = {
    [3584 + 0 * 72] = 11, [3584 + 1 * 72] = 22, [3584 + 2 * 72] = 33,
    [3584 + 3 * 72] = 44, [3584 + 4 * 72] = 55, [3584 + 5 * 72] = 66,
    [3584 + 6 * 72] = 77, [3584 + 7 * 72] = 88,
};

int main(void) {
  const uint32_t bank = 0;
  for (int test = 0; test < 3; ++test) {
    uintptr_t source = test ? (uintptr_t)(page_input + 3584) : (uintptr_t)input;
    int stride = test ? 72 : 256;
    int first_row = test == 2 ? 56 : 0;
    int rows = test == 2 ? 64 : 8;
    bb_mem_alloc(bank, 1, 1);
    if (test == 2)
      bb_mvin((uintptr_t)zeros, bank, 64, 1);
    bb_mvin_2d(source, bank, 1, stride, 8, first_row, 8, BANK_WIDTH / 8);
    bb_mvout((uintptr_t)output, bank, rows, 1);

    for (int row = 0; row < rows; ++row) {
      for (int byte = 0; byte < BANK_WIDTH / 8; ++byte) {
        uint8_t expected = row >= first_row && byte == 0
                               ? (uint8_t)((row - first_row + 1) * 11)
                               : 0;
        uint8_t actual = output[row * (BANK_WIDTH / 8) + byte];
        if (actual != expected) {
          printf("mvin_2d pixel stride mismatch case=%d row=%d byte=%d "
                 "expected=%u "
                 "got=%u\n",
                 test, row, byte, expected, actual);
          return 1;
        }
      }
    }
  }

  printf("mvin_2d pixel stride test PASSED\n");
  return 0;
}
