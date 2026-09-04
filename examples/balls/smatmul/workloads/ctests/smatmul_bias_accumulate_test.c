#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

static int8_t identity[16 * 16] __attribute__((aligned(64)));
static int8_t weight0[16 * 16] __attribute__((aligned(64)));
static int8_t weight1[16 * 16] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static int32_t actual[16 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < 16; ++row) {
    bias[row] = row - 8;
    for (int col = 0; col < 16; ++col) {
      identity[row * 16 + col] = row == col ? 1 : 0;
      weight0[row * 16 + col] = row + col;
      weight1[row * 16 + col] = row - col;
    }
  }

  for (int bank = 0; bank < 6; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)identity, 1, 16, 1);
  bb_mvin((uintptr_t)weight0, 2, 16, 1);
  bb_mvin((uintptr_t)identity, 3, 16, 1);
  bb_mvin((uintptr_t)weight1, 4, 16, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 5, 16, 16, 16, 1, 0, 0);
  bb_smatmul_os(3, 4, 5, 16, 16, 16, 0, 1, 0);
  bb_mvout((uintptr_t)actual, 5, 64, 1);
  bb_fence();

  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int expected = 2 * row + bias[col];
      int index = row * 16 + col;
      if (actual[index] != expected) {
        printf("smatmul_bias_accumulate FAIL row=%d col=%d expected=%d "
               "actual=%d\n",
               row, col, expected, actual[index]);
        return 1;
      }
    }
  }
  printf("smatmul_bias_accumulate PASS\n");
  return 0;
}
