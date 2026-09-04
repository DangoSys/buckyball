#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/quant.h>
#include <stdint.h>
#include <stdio.h>

static int32_t input[64] __attribute__((aligned(64)));
static float scales[16] __attribute__((aligned(64)));
static int8_t actual[192] __attribute__((aligned(64)));

int main(void) {
  for (int channel = 0; channel < 16; ++channel)
    scales[channel] = 1.0f;
  for (int position = 0; position < 4; ++position)
    for (int channel = 0; channel < 16; ++channel)
      input[position * 16 + channel] = position * 20 + channel - 8;

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 16, 1);
  bb_mvin((uintptr_t)scales, 1, 4, 1);
  bb_quant_i32_to_i8(0, 1, 2, 16, 0, 3, 2, 2, 4, 1);
  bb_mvout((uintptr_t)actual, 2, 12, 1);
  bb_fence();

  for (int row = 0; row < 4; ++row) {
    for (int channel = 0; channel < 16; ++channel) {
      int index = row * 16 + channel;
      int output_row = 3 + (row / 2) * 4 + row % 2;
      int actual_index = output_row * 16 + channel;
      int8_t expected = input[index] < 0 ? 0 : (int8_t)input[index];
      if (actual[actual_index] != expected) {
        printf("quant_i32_to_i8 FAIL row=%d channel=%d expected=%d actual=%d\n",
               row, channel, expected, actual[actual_index]);
        return 1;
      }
    }
  }
  for (int row = 0; row < 12; ++row) {
    if (row == 3 || row == 4 || row == 7 || row == 8)
      continue;
    for (int channel = 0; channel < 16; ++channel)
      if (actual[row * 16 + channel] != 0) {
        printf("quant_i32_to_i8 FAIL clobbered row=%d channel=%d\n", row,
               channel);
        return 1;
      }
  }
  printf("quant_i32_to_i8 PASS\n");
  return 0;
}
