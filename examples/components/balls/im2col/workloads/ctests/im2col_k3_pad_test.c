#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 3x3 kernel with padding 1 over a 2x2 input gives 2x2 windows. One output
 * group holds all nine taps in one 4-row M tile: tap t, window w is line
 * t * 4 + w, zero when the tap falls into the padding. */
static int8_t input[4 * 16] __attribute__((aligned(64)));
static int8_t actual[36 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 4 * 16; ++i)
    input[i] = (int8_t)(i * 7 + 3);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_im2col(0, 1, 2, 3, 1, 1, 0, 0, 0, 0, 4, 4);
  bb_mvout((uintptr_t)actual, 1, 36, 1);

  for (int tap = 0; tap < 9; ++tap)
    for (int w = 0; w < 4; ++w) {
      int y = w / 2 + tap / 3 - 1, x = w % 2 + tap % 3 - 1;
      int inside = y >= 0 && y < 2 && x >= 0 && x < 2;
      for (int lane = 0; lane < 16; ++lane) {
        int8_t expected = inside ? input[(y * 2 + x) * 16 + lane] : 0;
        int8_t got = actual[(tap * 4 + w) * 16 + lane];
        if (got != expected) {
          printf("im2col_k3_pad FAIL tap=%d window=%d lane=%d expected=%d "
                 "actual=%d\n",
                 tap, w, lane, expected, got);
          return 1;
        }
      }
    }
  printf("im2col_k3_pad PASS\n");
  return 0;
}
