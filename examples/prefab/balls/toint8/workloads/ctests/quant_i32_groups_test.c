#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/quant.h>
#include <stdint.h>
#include <stdio.h>

/* QUANT_I32_TO_I8 over two channel groups: group g of the input, scale, and
 * output banks is channel block g; every group uses the same output core. */
enum { GROUPS = 2, PIXELS = 4, ROWS = PIXELS * 4, OUT_ROWS = 6 };

static int32_t input[GROUPS][PIXELS * 16] __attribute__((aligned(64)));
static float scales[GROUPS][16] __attribute__((aligned(64)));
static int8_t actual[GROUPS][OUT_ROWS * 16] __attribute__((aligned(64)));

int main(void) {
  for (int group = 0; group < GROUPS; ++group) {
    for (int channel = 0; channel < 16; ++channel)
      scales[group][channel] = (group + channel) % 2 ? 2.0f : 1.0f;
    for (int i = 0; i < PIXELS * 16; ++i)
      input[group][i] = (group * 37 + i * 11) % 160 - 80;
  }

  bb_mem_alloc(0, 1, GROUPS);
  bb_mem_alloc(1, 1, GROUPS);
  bb_mem_alloc(2, 1, GROUPS);
  for (int group = 0; group < GROUPS; ++group) {
    bb_mvin_group((uintptr_t)input[group], 0, group, ROWS, 1);
    bb_mvin_group((uintptr_t)scales[group], 1, group, 4, 1);
  }
  // Output core: base 1, width 2, height 2, stride 3 -> rows 1, 2, 4, 5.
  bb_quant_i32_to_i8(0, 1, 2, ROWS, 0, 1, 2, 2, 3, 0);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvout_group((uintptr_t)actual[group], 2, group, OUT_ROWS, 1);

  for (int group = 0; group < GROUPS; ++group)
    for (int row = 0; row < OUT_ROWS; ++row)
      for (int channel = 0; channel < 16; ++channel) {
        int expected = 0;
        if (row % 3) {
          int pixel = row / 3 * 2 + row % 3 - 1;
          int value =
              input[group][pixel * 16 + channel] * (int)scales[group][channel];
          expected = value > 127 ? 127 : value < -128 ? -128 : value;
        }
        if (actual[group][row * 16 + channel] != expected) {
          printf("quant_i32_groups FAIL group=%d row=%d channel=%d "
                 "expected=%d actual=%d\n",
                 group, row, channel, expected,
                 actual[group][row * 16 + channel]);
          return 1;
        }
      }
  printf("quant_i32_groups PASS\n");
  return 0;
}
