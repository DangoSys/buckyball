#include "buckyball.h"
#include <dma.h>
#include <isa/maxpool.h>
#include <multithread.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 16, LANES = BANK_WIDTH / 8 };
static int8_t input[BB_CORE_NUM][2 * LANES] __attribute__((aligned(64)));
static int8_t initial[BB_CORE_NUM][ROWS * LANES] __attribute__((aligned(64)));
static int8_t output[BB_CORE_NUM][ROWS * LANES] __attribute__((aligned(64)));

extern "C" int pebble_pipeline(uint32_t instance) {
  int c = instance;
  for (int i = 0; i < 2 * LANES; ++i)
    input[c][i] = i - LANES;
  for (int i = 0; i < ROWS * LANES; ++i)
    initial[c][i] = -37;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input[c], 0, 2, 1);
  bb_mvin((uintptr_t)initial[c], 1, ROWS, 1);
  bb_maxpool(0, 1, 2, 1, 1, 1, 0, 0, 5, 1, 0, 0);
  bb_maxpool(0, 1, 2, 1, 1, 1, 0, 1, 11, 1, 0, 0);
  bb_mvout((uintptr_t)output[c], 1, ROWS, 1);
  for (int row = 0; row < ROWS; ++row)
    for (int lane = 0; lane < LANES; ++lane) {
      int8_t expected = row == 5    ? input[c][lane]
                        : row == 11 ? input[c][LANES + lane]
                                    : -37;
      if (output[c][row * LANES + lane] != expected) {
        printf("maxpool_partial_update FAIL row=%d lane=%d\n", row, lane);
        return 1;
      }
    }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("maxpool_partial_update PASS\n");
  return 0;
}
