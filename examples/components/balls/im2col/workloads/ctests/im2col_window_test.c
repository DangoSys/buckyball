#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 2x2 kernel over a 3x3 NHWC input with 32 channels (two channel groups) at
 * input_base 3, windows 1..2 of the 2x2 output. K tile t = tap * 2 + g; each
 * of the eight output groups holds one tile: line w is channels 16g..16g+15
 * of pixel (oy + kr, ox + kc) for window 1 + w. */
static int8_t input[(3 + 9) * 32] __attribute__((aligned(64)));
static int8_t actual[2 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 3 * 32; ++i)
    input[i] = 0x5a;
  for (int i = 3 * 32; i < 12 * 32; ++i)
    input[i] = (int8_t)(i * 7 - 9);

  bb_mem_alloc(0, 1, 2);
  bb_mem_alloc(1, 1, 8);
  bb_mvin((uintptr_t)input, 0, 12, 1);
  bb_im2col(0, 1, 3, 2, 1, 0, 3, 0, 0, 1, 2, 2);

  for (int tile = 0; tile < 8; ++tile) {
    int tap = tile / 2, group = tile % 2;
    bb_mvout_group((uintptr_t)actual, 1, tile, 2, 1);
    for (int w = 0; w < 2; ++w) {
      int window = 1 + w;
      int pixel = (window / 2 + tap / 2) * 3 + window % 2 + tap % 2;
      for (int lane = 0; lane < 16; ++lane) {
        int8_t expected = input[(3 + pixel) * 32 + group * 16 + lane];
        if (actual[w * 16 + lane] != expected) {
          printf("im2col_window FAIL tile=%d window=%d lane=%d expected=%d "
                 "actual=%d\n",
                 tile, w, lane, expected, actual[w * 16 + lane]);
          return 1;
        }
      }
    }
  }
  printf("im2col_window PASS\n");
  return 0;
}
