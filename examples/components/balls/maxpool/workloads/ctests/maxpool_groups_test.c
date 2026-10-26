#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

/* 2x2 stride-2 MAXPOOL over an NHWC 4x4 input with 32 channels: input and
 * output group g are channel block g, so one instruction pools both blocks
 * and a whole-bank DMA moves NHWC rows directly. */
enum { GROUPS = 2, CHANNELS = 16 * GROUPS, SIDE = 4, OUT = 2 };

static int8_t input[SIDE * SIDE * CHANNELS] __attribute__((aligned(64)));
static int8_t actual[OUT * OUT * CHANNELS] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < SIDE * SIDE * CHANNELS; ++i)
    input[i] = (int8_t)((i * 37 + 11) % 255 - 127);

  bb_mem_alloc(0, 1, GROUPS);
  bb_mem_alloc(1, 1, GROUPS);
  bb_mvin((uintptr_t)input, 0, SIDE * SIDE, 1);
  bb_maxpool(0, 1, SIDE, OUT, 2, 2, 0, 0, 0, OUT, 0, 0);
  bb_mvout((uintptr_t)actual, 1, OUT * OUT, 1);

  for (int oy = 0; oy < OUT; ++oy)
    for (int ox = 0; ox < OUT; ++ox)
      for (int channel = 0; channel < CHANNELS; ++channel) {
        int expected = -128;
        for (int ky = 0; ky < 2; ++ky)
          for (int kx = 0; kx < 2; ++kx) {
            int value = input[((oy * 2 + ky) * SIDE + ox * 2 + kx) * CHANNELS +
                              channel];
            expected = value > expected ? value : expected;
          }
        int got = actual[(oy * OUT + ox) * CHANNELS + channel];
        if (got != expected) {
          printf("maxpool_groups FAIL y=%d x=%d channel=%d expected=%d "
                 "actual=%d\n",
                 oy, ox, channel, expected, got);
          return 1;
        }
      }
  bb_mem_release(0);
  bb_mem_release(1);
  printf("maxpool_groups PASS\n");
  return 0;
}
