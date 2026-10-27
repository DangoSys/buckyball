#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

/* 1x1 kernel over a 2x2 NHWC input with 32 channels (two channel groups) and
 * two output groups: output group g is K tile g, so the output, read back as
 * a whole two-group bank, is the input itself. */
static int8_t input[4 * 32] __attribute__((aligned(64)));
static int8_t actual[4 * 32] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 4 * 32; ++i)
    input[i] = (int8_t)(i * 3 - 100);

  bb_mem_alloc(0, 1, 2);
  bb_mem_alloc(1, 1, 2);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_im2col(0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 4, 4);
  bb_mvout((uintptr_t)actual, 1, 4, 1);

  for (int i = 0; i < 4 * 32; ++i)
    if (actual[i] != input[i]) {
      printf("im2col_k1 FAIL index=%d expected=%d actual=%d\n", i, input[i],
             actual[i]);
      return 1;
    }
  printf("im2col_k1 PASS\n");
  return 0;
}
