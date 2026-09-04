#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <stdint.h>
#include <stdio.h>

enum { INPUT_BASE = 5, LANE = 7 };

static int8_t input[64 * 16] __attribute__((aligned(64)));
static int8_t actual[16 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 64 * 16; ++i)
    input[i] = -99;
  for (int i = 0; i < 36; ++i)
    input[(INPUT_BASE + i) * 16 + LANE] = i + 1;

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 64, 1);
  bb_im2col(0, 1, 6, 3, 1, 0, INPUT_BASE, LANE, 0, 0, 4, 8);
  bb_mvout((uintptr_t)actual, 1, 16, 1);
  bb_fence();

  for (int local = 0; local < 8; ++local) {
    int global = local + 4;
    int output_row = global / 4;
    int output_col = global % 4;
    for (int kr = 0; kr < 3; ++kr) {
      for (int kc = 0; kc < 3; ++kc) {
        int expected =
            input[(INPUT_BASE + (output_row + kr) * 6 + output_col + kc) * 16 +
                  LANE];
        int index = local * 16 + kr * 3 + kc;
        if (actual[index] != expected) {
          printf("im2col_window FAIL local=%d k=%d expected=%d actual=%d\n",
                 local, kr * 3 + kc, expected, actual[index]);
          return 1;
        }
      }
    }
  }
  printf("im2col_window PASS\n");
  return 0;
}
