#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

enum {
  HEIGHT = 1,
  TILE_WIDTH = 8,
  PIXEL_BYTES = 256,
  SOURCE_WIDTH = 8,
  VALID_BYTES = 16,
};

static const uint8_t input[SOURCE_WIDTH * PIXEL_BYTES]
    __attribute__((aligned(128))) = {
        [0 * PIXEL_BYTES] = 11, [1 * PIXEL_BYTES] = 22, [2 * PIXEL_BYTES] = 33,
        [3 * PIXEL_BYTES] = 44, [4 * PIXEL_BYTES] = 55, [5 * PIXEL_BYTES] = 66,
        [6 * PIXEL_BYTES] = 77, [7 * PIXEL_BYTES] = 88,
};
static uint8_t output[TILE_WIDTH * VALID_BYTES] __attribute__((aligned(128)));

int main(void) {
  const uint32_t bank = 0;
  bb_mem_alloc(bank, 1, 1);
  bb_mvin_2d((uintptr_t)input, bank, HEIGHT, PIXEL_BYTES, SOURCE_WIDTH, 0,
             TILE_WIDTH, VALID_BYTES);
  bb_mvout((uintptr_t)output, bank, TILE_WIDTH, 1);
  bb_fence();

  for (int pixel = 0; pixel < TILE_WIDTH; ++pixel) {
    for (int byte = 0; byte < VALID_BYTES; ++byte) {
      uint8_t expected = byte == 0 ? (uint8_t)((pixel + 1) * 11) : 0;
      uint8_t actual = output[pixel * VALID_BYTES + byte];
      if (actual != expected) {
        printf("mvin_2d pixel stride mismatch pixel=%d byte=%d expected=%u "
               "got=%u\n",
               pixel, byte, expected, actual);
        return 1;
      }
    }
  }

  printf("mvin_2d pixel stride test PASSED\n");
  return 0;
}
