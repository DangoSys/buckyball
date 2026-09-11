#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

enum { LANES = BANK_WIDTH / 8, INPUT_ROWS = 36, OUTPUT_ROWS = 9 };
static int8_t input[INPUT_ROWS * LANES] __attribute__((aligned(64)));
static int8_t output[OUTPUT_ROWS * LANES] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < INPUT_ROWS; ++row)
    for (int lane = 0; lane < LANES; ++lane)
      input[row * LANES + lane] = (int8_t)(row + lane - 32);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, INPUT_ROWS, 1);
  bb_maxpool(0, 1, 6, 3, 2, 2, 0, 0, 0, 3, 0, 0);
  bb_im2col(1, 2, 3, 1, 1, 0, 0, 0, 0, 0, 0, OUTPUT_ROWS);
  bb_mvout((uintptr_t)output, 2, OUTPUT_ROWS, 1);
  bb_fence();
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
      int source = (y * 2 + 1) * 6 + x * 2 + 1;
      int window = y * 3 + x;
      if (output[window * LANES] != input[source * LANES]) {
        printf("maxpool im2col mismatch y=%d x=%d\n", y, x);
        return 1;
      }
    }
  printf("maxpool im2col PASS\n");
  return 0;
}
