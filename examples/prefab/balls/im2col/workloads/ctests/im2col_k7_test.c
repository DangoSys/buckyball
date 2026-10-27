#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 7x7 stride-2 kernel over a 9x9 input gives 2x2 windows. Seven output groups
 * hold one kernel row each (seven taps): group kr, tap kc, window w is line
 * kc * 4 + w, the input row of pixel (oy * 2 + kr, ox * 2 + kc). */
static int8_t input[81 * 16] __attribute__((aligned(64)));
static int8_t actual[28 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 81 * 16; ++i)
    input[i] = (int8_t)(i * 3 + 11);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 7);
  bb_mvin((uintptr_t)input, 0, 81, 1);
  bb_im2col(0, 1, 9, 7, 2, 0, 0, 0, 0, 0, 4, 4);

  for (int kr = 0; kr < 7; ++kr) {
    bb_mvout_group((uintptr_t)actual, 1, kr, 28, 1);
    for (int kc = 0; kc < 7; ++kc)
      for (int w = 0; w < 4; ++w) {
        int pixel = (w / 2 * 2 + kr) * 9 + w % 2 * 2 + kc;
        for (int lane = 0; lane < 16; ++lane) {
          int8_t expected = input[pixel * 16 + lane];
          int8_t got = actual[(kc * 4 + w) * 16 + lane];
          if (got != expected) {
            printf("im2col_k7 FAIL kr=%d kc=%d window=%d lane=%d expected=%d "
                   "actual=%d\n",
                   kr, kc, w, lane, expected, got);
            return 1;
          }
        }
      }
  }
  printf("im2col_k7 PASS\n");
  return 0;
}
