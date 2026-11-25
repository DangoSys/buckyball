#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(THREAD_MEM_BANK_WIDTH == 128 && THREAD_MEM_BANK_ENTRIES >= 64,
               "Strided DMA requires 128-bit bank rows and at least 64 rows");
static const uint8_t input[8192] __attribute__((aligned(4096))) = {
    [4090] = 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
    [4110] = 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    [4130] = 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    [4150] = 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
};
static uint8_t banks[13 * 32] __attribute__((aligned(64))) = {0xa5};
static uint8_t output[256] __attribute__((aligned(64)));

int main(void) {
  const unsigned first = THREAD_MEM_BANK_ENTRIES - 16;
  memset(output, 0xa5, sizeof(output));
  bb_mset_clear(0, 1, 2);
  // Partial rows, an unaligned page crossing, a selected group and a bank
  // offset beyond the old 2D encoding. Skipped bank rows must remain zero.
  bb_mvin_strided((uintptr_t)(input + 4090), 0, 1, first, 3, 4, 20, 12, 0);
  bb_mvout_strided((uintptr_t)banks, 0, 0, first - 1, 1, 13, 32, 16, 0);
  bb_mvout_strided((uintptr_t)(banks + 16), 0, 1, first - 1, 1, 13, 32, 16, 0);
  for (unsigned row = first - 1; row < first + 12; ++row) {
    for (unsigned group = 0; group < 2; ++group) {
      for (unsigned byte = 0; byte < 16; ++byte) {
        unsigned lane = row >= first ? row - first : 16;
        uint8_t expected = group == 1 && lane < 12 && lane % 3 == 0 && byte < 12
                               ? input[4090 + lane / 3 * 20 + byte]
                               : 0;
        if (banks[(row - first + 1) * 32 + group * 16 + byte] != expected) {
          printf("Strided read mismatch: row=%u group=%u byte=%u\n", row, group,
                 byte);
          return 1;
        }
      }
    }
  }
  // Only seven bytes of each DDR row may change; the gaps remain sentinel.
  bb_mvout_strided((uintptr_t)(output + 5), 0, 1, first, 3, 4, 23, 7, 0);
  for (unsigned i = 0; i < sizeof(output); ++i) {
    unsigned at = i >= 5 ? i - 5 : 256;
    uint8_t expected = at / 23 < 4 && at % 23 < 7
                           ? input[4090 + at / 23 * 20 + at % 23]
                           : 0xa5;
    if (output[i] != expected) {
      printf("Strided write mismatch: byte=%u\n", i);
      return 1;
    }
  }
  bb_mem_release(0);
  printf("Strided DMA PASSED\n");
  return 0;
}
