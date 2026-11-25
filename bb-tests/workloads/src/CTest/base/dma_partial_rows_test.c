#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

static const uint8_t input[64] __attribute__((aligned(64))) = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
static uint8_t output[64] __attribute__((aligned(64))) = {[0 ... 63] = 0xa5};

int main(void) {
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 1, 4, 1);
  bb_mvout_strided((uintptr_t)output, 1, 0, 0, 1, 2, 20, 16, 0);
  bb_mvout_strided((uintptr_t)(output + 16), 1, 0, 2, 1, 2, 20, 4, 0);
  bb_mem_release(1);
  for (unsigned i = 0; i < sizeof(output); ++i) {
    unsigned row = i / 20, column = i % 20;
    uint8_t expected = row >= 2      ? 0xa5
                       : column < 16 ? input[row * 16 + column]
                                     : input[32 + row * 16 + column - 16];
    if (output[i] != expected) {
      printf("Partial DMA mismatch at %u: %u != %u\n", i, output[i], expected);
      return 1;
    }
  }
  return 0;
}
