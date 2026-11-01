#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/quant.h>
#include <stdint.h>
#include <stdio.h>

/* QUANT_F32_TO_I8 with one multiplier per channel: channel c of block g is
 * lane c % 4 of input row 4 * pixel + c / 4 and takes lane c % 4 of row
 * 4g + c / 4 of the one scale group. */
enum { GROUPS = 2, PIXELS = 3, ROWS = PIXELS * 4 };

static float input[GROUPS][PIXELS * 16] __attribute__((aligned(64)));
static float scales[GROUPS][16] __attribute__((aligned(64)));
static int8_t actual[GROUPS][PIXELS * 16] __attribute__((aligned(64)));

int main(void) {
  for (int group = 0; group < GROUPS; ++group) {
    for (int i = 0; i < PIXELS * 16; ++i)
      input[group][i] = (float)((group * 13 + i * 5) % 61 - 30);
    for (int c = 0; c < 16; ++c)
      scales[group][c] = (float)(1 + (group * 16 + c) % 5);
  }

  bb_mem_alloc(0, 1, GROUPS);
  bb_mem_alloc(1, 1, GROUPS);
  bb_mem_alloc(2, 1, 1);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvin_group((uintptr_t)input[group], 0, group, ROWS, 1);
  // One scale group: rows 4g..4g+3 hold the scales of channel block g.
  bb_mvin((uintptr_t)scales, 2, GROUPS * 4, 1);
  bb_quant_f32_to_i8(0, 2, 1, ROWS);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvout_group((uintptr_t)actual[group], 1, group, PIXELS, 1);

  for (int group = 0; group < GROUPS; ++group)
    for (int i = 0; i < PIXELS * 16; ++i) {
      int value = (int)input[group][i] * (int)scales[group][i % 16];
      int expected = value > 127 ? 127 : value < -128 ? -128 : value;
      if (actual[group][i] != expected) {
        printf("quant_f32_channels FAIL group=%d index=%d expected=%d "
               "actual=%d\n",
               group, i, expected, actual[group][i]);
        return 1;
      }
    }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("quant_f32_channels PASS\n");
  return 0;
}
