#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* Window range 5..7 of a 1x1 im2col over a 4x4 input: lines 0..2 are pixels
 * 5, 6, 7 and lines 3..7 zero-fill the partial 8-row M tile. */
static int8_t input[16 * 16] __attribute__((aligned(64)));
static int8_t actual[8 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 16 * 16; ++i)
    input[i] = (int8_t)(i * 11 + 7);
  for (int i = 0; i < 8 * 16; ++i)
    actual[i] = 99;

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 16, 1);
  bb_im2col(0, 1, 4, 1, 1, 0, 0, 0, 0, 5, 3, 8);
  bb_mvout((uintptr_t)actual, 1, 8, 1);

  for (int line = 0; line < 8; ++line)
    for (int lane = 0; lane < 16; ++lane) {
      int8_t expected = line < 3 ? input[(5 + line) * 16 + lane] : 0;
      if (actual[line * 16 + lane] != expected) {
        printf("im2col_k3_partial FAIL line=%d lane=%d expected=%d "
               "actual=%d\n",
               line, lane, expected, actual[line * 16 + lane]);
        return 1;
      }
    }
  printf("im2col_k3_partial PASS\n");
  return 0;
}
