#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/int2fp.h>
#include <stdint.h>
#include <stdio.h>

/* INT32_TO_FP32 with ReLU over two channel groups: group g of the input,
 * scale, and output banks is channel block g; every group uses the same rows.
 */
enum { GROUPS = 2, ROWS = 8 };

static int32_t input[GROUPS][ROWS * 4] __attribute__((aligned(64)));
static float scales[GROUPS][16] __attribute__((aligned(64)));
static float output[GROUPS][ROWS * 4] __attribute__((aligned(64)));

int main(void) {
  for (int group = 0; group < GROUPS; ++group) {
    for (int i = 0; i < 16; ++i)
      scales[group][i] = (float)(1 << ((group + i) % 3)) * 0.5f;
    for (int i = 0; i < ROWS * 4; ++i)
      input[group][i] = (group * 13 + i * 5) % 40 - 15;
  }

  bb_mem_alloc(0, 1, GROUPS);
  bb_mem_alloc(1, 1, GROUPS);
  bb_mem_alloc(2, 1, GROUPS);
  for (int group = 0; group < GROUPS; ++group) {
    bb_mvin_group((uintptr_t)input[group], 0, group, ROWS, 1);
    bb_mvin_group((uintptr_t)scales[group], 1, group, 4, 1);
  }
  bb_int32_to_fp32(0, 1, 2, ROWS, 1);
  for (int group = 0; group < GROUPS; ++group)
    bb_mvout_group((uintptr_t)output[group], 2, group, ROWS, 1);

  for (int group = 0; group < GROUPS; ++group)
    for (int i = 0; i < ROWS * 4; ++i) {
      int value = input[group][i] < 0 ? 0 : input[group][i];
      float expected = (float)value * scales[group][i % 16];
      if (output[group][i] != expected) {
        printf("int2fp_groups FAIL group=%d index=%d\n", group, i);
        return 1;
      }
    }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("int2fp_groups PASS\n");
  return 0;
}
