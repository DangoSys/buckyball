#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
_Static_assert(THREAD_MEM_BANK_WIDTH == 128 && THREAD_MEM_BANK_ENTRIES >= 24,
               "Blocked DMA requires 128-bit rows and at least 24 bank rows");
// clang-format off
static const uint8_t input[8192] __attribute__((aligned(4096))) = {
    [4090] = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    [4170] = 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    [4250] = 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
};
// clang-format on
static uint8_t bank_data[24 * 32] __attribute__((aligned(64)));
static uint8_t output[320] __attribute__((aligned(64)));
int main(void) {
  memset(output, 0xa5, sizeof(output));
  bb_mset_clear(0, 1, 2);
  // Three two-row blocks, crossing a page and leaving gaps in bank group 1.
  bb_mvin_strided((uintptr_t)(input + 4090), 0, 1, 3, 4, 6, 80, 16, 1);
  bb_mvout((uintptr_t)bank_data, 0, 24, 1);
  for (unsigned r = 0; r < 24; ++r)
    for (unsigned g = 0; g < 2; ++g)
      for (unsigned b = 0; b < 16; ++b) {
        unsigned at = r >= 3 ? r - 3 : 24;
        uint8_t expected = g == 1 && at / 4 < 3 && at % 4 < 2
                               ? input[4090 + at / 4 * 80 + at % 4 * 16 + b]
                               : 0;
        if (bank_data[r * 32 + g * 16 + b] != expected)
          return 1;
      }
  bb_mvout_strided((uintptr_t)(output + 7), 0, 1, 3, 4, 6, 32, 16, 1);
  for (unsigned i = 0; i < sizeof(output); ++i) {
    unsigned at = i >= 7 ? i - 7 : 320;
    uint8_t expected = at / 32 < 3 && at % 32 < 32
                           ? input[4090 + at / 32 * 80 + at % 32]
                           : 0xa5;
    if (output[i] != expected)
      return 1;
  }
  bb_mem_release(0);
  printf("Strided two-row blocks PASSED\n");
  return 0;
}
