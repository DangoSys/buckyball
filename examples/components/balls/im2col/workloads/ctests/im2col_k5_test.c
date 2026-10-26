#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 5x5 kernel over a 5x5 input: one window, 25 taps. With tile_rows 1 and five
 * output groups, group kr line kc is the input row of pixel (kr, kc), so each
 * output group is one input image row. */
static int8_t input[25 * 16] __attribute__((aligned(64)));
static int8_t actual[5 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 25 * 16; ++i)
    input[i] = (int8_t)(i * 13 - 7);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 5);
  bb_mvin((uintptr_t)input, 0, 25, 1);
  bb_im2col(0, 1, 5, 5, 1, 0, 0, 0, 0, 0, 1, 1);

  for (int kr = 0; kr < 5; ++kr) {
    bb_mvout_group((uintptr_t)actual, 1, kr, 5, 1);
    for (int i = 0; i < 5 * 16; ++i)
      if (actual[i] != input[kr * 5 * 16 + i]) {
        printf("im2col_k5 FAIL row=%d index=%d expected=%d actual=%d\n", kr, i,
               input[kr * 5 * 16 + i], actual[i]);
        return 1;
      }
  }
  printf("im2col_k5 PASS\n");
  return 0;
}
