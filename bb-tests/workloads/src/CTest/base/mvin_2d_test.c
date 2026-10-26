#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <params.h>
#include <stdio.h>

enum {
  SOURCE_HEIGHT = 5,
  SOURCE_WIDTH = 7,
  PANEL_BYTES = (BANK_WIDTH / 8),
  CHANNELS = (PANEL_BYTES + PANEL_BYTES / 2),
  CORE_HEIGHT = 4,
  CORE_WIDTH = 3,
  START_Y = 1,
  START_X = 2,
  CORE_ROWS = (CORE_HEIGHT * CORE_WIDTH)
};

static elem_t input[SOURCE_HEIGHT * SOURCE_WIDTH * CHANNELS]
    __attribute__((aligned(128)));
static elem_t output[2 * CORE_ROWS * PANEL_BYTES] __attribute__((aligned(128)));
static elem_t dirty[2 * CORE_ROWS * PANEL_BYTES] __attribute__((aligned(128)));

static int check_output(void) {
  for (int panel = 0; panel < 2; ++panel) {
    for (int y = 0; y < CORE_HEIGHT; ++y) {
      for (int x = 0; x < CORE_WIDTH; ++x) {
        for (int lane = 0; lane < PANEL_BYTES; ++lane) {
          int channel = panel * PANEL_BYTES + lane;
          elem_t expected =
              channel < CHANNELS
                  ? input[((START_Y + y) * SOURCE_WIDTH + (START_X + x)) *
                              CHANNELS +
                          channel]
                  : 0;
          int row = panel * CORE_ROWS + y * CORE_WIDTH + x;
          elem_t actual = output[row * PANEL_BYTES + lane];
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
  for (int i = 0; i < 2 * CORE_ROWS * PANEL_BYTES; ++i) {
    output[i] = (elem_t)-1;
    dirty[i] = (elem_t)0x5a;
  }

  uint32_t bank = 0;
  bb_mem_alloc(bank, 1, 1);
  // Tail lanes must be zeroed even when the destination already contains data.
  bb_mvin((uintptr_t)dirty, bank, 2 * CORE_ROWS, 1);
  uintptr_t source =
      (uintptr_t)&input[(START_Y * SOURCE_WIDTH + START_X) * CHANNELS];
  bb_mvin_2d(source, bank, CORE_HEIGHT, CHANNELS, SOURCE_WIDTH, 0, CORE_WIDTH,
             PANEL_BYTES);
  bb_mvin_2d(source + PANEL_BYTES, bank, CORE_HEIGHT, CHANNELS, SOURCE_WIDTH,
             CORE_ROWS, CORE_WIDTH, CHANNELS - PANEL_BYTES);
  bb_mvout((uintptr_t)output, bank, 2 * CORE_ROWS, 1);

  if (!check_output())
    return 1;
  printf("mvin_2d test PASSED\n");
  return 0;
}
