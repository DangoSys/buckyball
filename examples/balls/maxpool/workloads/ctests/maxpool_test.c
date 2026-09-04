#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

enum { INPUT_BASE = 7, OUTPUT_BASE = 11, OUTPUT_STRIDE = 7 };

static int8_t input[64 * 16] __attribute__((aligned(64)));
static int8_t output[64 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 64 * 16; ++i) {
    input[i] = 42;
    output[i] = 23;
  }
  for (int position = 0; position < 6 * 6; ++position)
    for (int channel = 0; channel < 16; ++channel)
      input[(INPUT_BASE + position) * 16 + channel] =
          (int8_t)(((position * 37 + channel * 19) & 255) - 128);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 64, 1);
  bb_mvin((uintptr_t)output, 1, 64, 1);
  bb_maxpool(0, 1, 6, 3, 2, 2, 0, INPUT_BASE, OUTPUT_BASE, OUTPUT_STRIDE, 0, 0);
  bb_mvout((uintptr_t)output, 1, 64, 1);
  bb_fence();

  for (int output_y = 0; output_y < 3; ++output_y) {
    for (int output_x = 0; output_x < 3; ++output_x) {
      for (int channel = 0; channel < 16; ++channel) {
        int8_t expected = -128;
        for (int kernel_y = 0; kernel_y < 2; ++kernel_y)
          for (int kernel_x = 0; kernel_x < 2; ++kernel_x) {
            int input_y = output_y * 2 + kernel_y;
            int input_x = output_x * 2 + kernel_x;
            int8_t value =
                input[(INPUT_BASE + input_y * 6 + input_x) * 16 + channel];
            if (value > expected)
              expected = value;
          }
        int offset =
            (OUTPUT_BASE + output_y * OUTPUT_STRIDE + output_x) * 16 + channel;
        if (output[offset] != expected) {
          printf("maxpool FAIL y=%d x=%d c=%d expected=%d actual=%d\n",
                 output_y, output_x, channel, expected, output[offset]);
          return 1;
        }
      }
    }
  }
  for (int row = 0; row < 64; ++row) {
    int selected = row >= OUTPUT_BASE &&
                   row <= OUTPUT_BASE + 2 * OUTPUT_STRIDE + 2 &&
                   (row - OUTPUT_BASE) % OUTPUT_STRIDE < 3;
    if (!selected)
      for (int channel = 0; channel < 16; ++channel)
        if (output[row * 16 + channel] != 23) {
          printf("maxpool clobbered row=%d channel=%d\n", row, channel);
          return 1;
        }
  }
  printf("maxpool PASS\n");
  return 0;
}
