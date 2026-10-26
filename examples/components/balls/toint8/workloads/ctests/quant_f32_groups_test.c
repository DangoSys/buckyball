#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/quant.h>
#include <stdint.h>
#include <stdio.h>

/* QUANT_F32_TO_I8 over three channel groups: input and output group g are
 * channel block g; every group uses the same rows. */
enum { GROUPS = 3, PIXELS = 2, ROWS = PIXELS * 4 };

static float input[GROUPS][PIXELS * 16] __attribute__((aligned(64)));
static int8_t actual[GROUPS][PIXELS * 16] __attribute__((aligned(64)));

int main(void) {
  for (int group = 0; group < GROUPS; ++group)
    for (int i = 0; i < PIXELS * 16; ++i)
      input[group][i] = (float)((group * 29 + i * 7) % 100 - 50);

  bb_mem_alloc(0, 1, GROUPS);
  bb_mem_alloc(1, 1, GROUPS);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvin_group((uintptr_t)input[group], 0, group, ROWS, 1);
  bb_quant_f32_to_i8(0, 1, ROWS, 2.0f);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvout_group((uintptr_t)actual[group], 1, group, PIXELS, 1);

  for (int group = 0; group < GROUPS; ++group)
    for (int i = 0; i < PIXELS * 16; ++i) {
      int value = (int)input[group][i] * 2;
      int expected = value > 127 ? 127 : value < -128 ? -128 : value;
      if (actual[group][i] != expected) {
        printf("quant_f32_groups FAIL group=%d index=%d expected=%d "
               "actual=%d\n",
               group, i, expected, actual[group][i]);
        return 1;
      }
    }
  printf("quant_f32_groups PASS\n");
  return 0;
}
