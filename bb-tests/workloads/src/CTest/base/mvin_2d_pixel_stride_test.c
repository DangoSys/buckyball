#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>

enum {
  HEIGHT = 1,
  TILE_WIDTH = 8,
  PIXEL_BYTES = 256,
  SOURCE_WIDTH = 8,
  VALID_BYTES = BANK_WIDTH / 8,
};

static const uint8_t input[SOURCE_WIDTH * PIXEL_BYTES]
    __attribute__((aligned(128))) = {
        [0 * PIXEL_BYTES] = 11, [1 * PIXEL_BYTES] = 22, [2 * PIXEL_BYTES] = 33,
        [3 * PIXEL_BYTES] = 44, [4 * PIXEL_BYTES] = 55, [5 * PIXEL_BYTES] = 66,
        [6 * PIXEL_BYTES] = 77, [7 * PIXEL_BYTES] = 88,
};
static uint8_t output[TILE_WIDTH * VALID_BYTES] __attribute__((aligned(128)));

// MobileNet's 72-byte pixels: the last read straddles the 4 KiB page.
static const uint8_t page_input[8192] __attribute__((aligned(4096))) = {
    [3584 + 0 * 72] = 11, [3584 + 1 * 72] = 22, [3584 + 2 * 72] = 33,
    [3584 + 3 * 72] = 44, [3584 + 4 * 72] = 55, [3584 + 5 * 72] = 66,
    [3584 + 6 * 72] = 77, [3584 + 7 * 72] = 88,
};

int main(void) {
  const uint32_t bank = 0;
  for (int test = 0; test < 2; ++test) {
    uintptr_t source = test ? (uintptr_t)(page_input + 3584) : (uintptr_t)input;
    int stride = test ? 72 : PIXEL_BYTES;
    bb_dma_fence();
    bb_mem_alloc(bank, 1, 1);
    bb_mvin_2d(source, bank, HEIGHT, stride, SOURCE_WIDTH, 0, TILE_WIDTH,
               VALID_BYTES);
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
  }

  printf("mvin_2d pixel stride test PASSED\n");
  return 0;
}
