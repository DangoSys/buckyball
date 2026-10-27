#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 3x3 kernel over a 4x4 input (16 channels) gives 2x2 windows. Nine output
 * groups hold one tap each: group kr * 3 + kc, line w is the input row of
 * pixel (oy + kr, ox + kc); lines 4..7 pad the 8-row M tile with zeros. */
static int8_t input[16 * 16] __attribute__((aligned(64)));
static int8_t actual[8 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 16 * 16; ++i)
    input[i] = (int8_t)(i * 5 + 1);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 9);
  bb_mvin((uintptr_t)input, 0, 16, 1);
  bb_im2col(0, 1, 4, 3, 1, 0, 0, 0, 0, 0, 4, 8);

  for (int tap = 0; tap < 9; ++tap) {
    bb_mvout_group((uintptr_t)actual, 1, tap, 8, 1);
    for (int w = 0; w < 8; ++w) {
      int pixel = (w / 2 + tap / 3) * 4 + w % 2 + tap % 3;
      for (int lane = 0; lane < 16; ++lane) {
        int8_t expected = w < 4 ? input[pixel * 16 + lane] : 0;
        if (actual[w * 16 + lane] != expected) {
          printf("im2col_k3 FAIL tap=%d window=%d lane=%d expected=%d "
                 "actual=%d\n",
                 tap, w, lane, expected, actual[w * 16 + lane]);
          return 1;
        }
      }
    }
  }
  printf("im2col_k3 PASS\n");
  return 0;
}
