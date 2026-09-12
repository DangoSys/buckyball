#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define SOURCE_HEIGHT 5
#define SOURCE_WIDTH 7
#define CHANNELS 24
#define TILE_HEIGHT 4
#define TILE_WIDTH 3
#define START_Y 1
#define START_X 2
#define TILE_ROWS (TILE_HEIGHT * TILE_WIDTH)

static elem_t input[SOURCE_HEIGHT * SOURCE_WIDTH * CHANNELS]
    __attribute__((aligned(128)));
static elem_t output[2 * TILE_ROWS * 16] __attribute__((aligned(128)));

static int check_output(void) {
  for (int panel = 0; panel < 2; ++panel) {
    for (int y = 0; y < TILE_HEIGHT; ++y) {
      for (int x = 0; x < TILE_WIDTH; ++x) {
        for (int lane = 0; lane < 16; ++lane) {
          int channel = panel * 16 + lane;
          elem_t expected =
              channel < CHANNELS
                  ? input[((START_Y + y) * SOURCE_WIDTH + (START_X + x)) *
                              CHANNELS +
                          channel]
                  : 0;
          int row = panel * TILE_ROWS + y * TILE_WIDTH + x;
          elem_t actual = output[row * 16 + lane];
          if (actual != expected) {
            printf("mvin_2d mismatch panel=%d y=%d x=%d lane=%d expected=%d "
                   "got=%d\n",
                   panel, y, x, lane, expected, actual);
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

int main(void) {
  for (int i = 0; i < SOURCE_HEIGHT * SOURCE_WIDTH * CHANNELS; ++i)
    input[i] = (elem_t)((i * 13 + 7) & 0x7f);
  for (int i = 0; i < 2 * TILE_ROWS * 16; ++i)
    output[i] = (elem_t)-1;

  uint32_t bank = 0;
  bb_mem_alloc(bank, 1, 1);
  uintptr_t source =
      (uintptr_t)&input[(START_Y * SOURCE_WIDTH + START_X) * CHANNELS];
  bb_mvin_2d(source, bank, TILE_HEIGHT, CHANNELS, SOURCE_WIDTH, 0, TILE_WIDTH,
             16);
  bb_mvin_2d(source + 16, bank, TILE_HEIGHT, CHANNELS, SOURCE_WIDTH, TILE_ROWS,
             TILE_WIDTH, CHANNELS - 16);
  bb_mvout((uintptr_t)output, bank, 2 * TILE_ROWS, 1);
  bb_fence();

  if (!check_output())
    return 1;
  printf("mvin_2d test PASSED\n");
  return 0;
}
