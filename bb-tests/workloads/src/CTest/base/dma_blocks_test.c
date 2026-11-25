#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
_Static_assert(THREAD_MEM_BANK_WIDTH == 128 && THREAD_MEM_BANK_ENTRIES >= 24,
               "Block DMA requires 128-bit rows and at least 24 bank rows");
// Three four-row blocks 288 rows (4608 bytes) apart in DDR: past the 4095-byte
// stride of the strided DMA.
#define STRIDE 288
static uint8_t input[3 * STRIDE * 16] __attribute__((aligned(4096)));
static uint8_t bank_data[24 * 32] __attribute__((aligned(64)));
static uint8_t output[3 * STRIDE * 16] __attribute__((aligned(4096)));
int main(void) {
  for (unsigned i = 0; i < sizeof(input); ++i)
    input[i] = (uint8_t)(i * 7 + i / 4608);
  memset(output, 0xa5, sizeof(output));
  bb_mset_clear(0, 1, 2);
  // Rows 3.. of bank group 1, six apart per block.
  bb_mvin_blocks((uintptr_t)input, 0, 1, 3, 6, 12, STRIDE, 2);
  bb_mvout((uintptr_t)bank_data, 0, 24, 1);
  for (unsigned r = 0; r < 24; ++r)
    for (unsigned g = 0; g < 2; ++g)
      for (unsigned b = 0; b < 16; ++b) {
        unsigned at = r >= 3 ? r - 3 : 24;
        uint8_t expected = g == 1 && at / 6 < 3 && at % 6 < 4
                               ? input[at / 6 * STRIDE * 16 + at % 6 * 16 + b]
                               : 0;
        if (bank_data[r * 32 + g * 16 + b] != expected)
          return 1;
      }
  bb_mvout_blocks((uintptr_t)output, 0, 1, 3, 6, 12, STRIDE, 2);
  for (unsigned i = 0; i < sizeof(output); ++i) {
    uint8_t expected = i % (STRIDE * 16) < 64 ? input[i] : 0xa5;
    if (output[i] != expected)
      return 1;
  }
  bb_mem_release(0);
  printf("Block DMA PASSED\n");
  return 0;
}
