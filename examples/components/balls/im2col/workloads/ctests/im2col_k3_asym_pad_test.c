#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* Bottom/right-only padding: 2x2 kernel, padding 1, start_row = start_col = 1
 * over a 3x3 input gives 3x3 windows reading pixel (oy + kr, ox + kc), zero
 * past the bottom/right edge. Four output groups hold one tap each; lines
 * 9..15 pad the 16-row M tile. */
static int8_t input[9 * 16] __attribute__((aligned(64)));
static int8_t actual[16 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 9 * 16; ++i)
    input[i] = (int8_t)(i * 9 - 50);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mvin((uintptr_t)input, 0, 9, 1);
  bb_im2col(0, 1, 3, 2, 1, 1, 0, 1, 1, 0, 9, 16);

  for (int tap = 0; tap < 4; ++tap) {
    bb_mvout_group((uintptr_t)actual, 1, tap, 16, 1);
    for (int w = 0; w < 16; ++w) {
      int y = w / 3 + tap / 2, x = w % 3 + tap % 2;
      int inside = w < 9 && y < 3 && x < 3;
      for (int lane = 0; lane < 16; ++lane) {
        int8_t expected = inside ? input[(y * 3 + x) * 16 + lane] : 0;
        if (actual[w * 16 + lane] != expected) {
          printf("im2col_k3_asym_pad FAIL tap=%d window=%d lane=%d "
                 "expected=%d actual=%d\n",
                 tap, w, lane, expected, actual[w * 16 + lane]);
          return 1;
        }
      }
    }
  }
  printf("im2col_k3_asym_pad PASS\n");
  return 0;
}
