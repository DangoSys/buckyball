#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

#define K 64

static int8_t input[K] __attribute__((aligned(64)));
static int8_t weight[K * 16] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static int32_t actual[16] __attribute__((aligned(64)));

int main(void) {
  for (int k = 0; k < K; ++k)
    input[k] = k % 7 - 3;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < 16; ++col)
      weight[k * 16 + col] = (k + col) % 11 - 5;
  for (int col = 0; col < 16; ++col)
    bias[col] = 3 * col - 8;

  for (int bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)input, 1, K / 16, 1);
  bb_mvin((uintptr_t)weight, 2, K, 1);
  bb_smatmul_bias(0);
  bb_smatmul_os(1, 2, 3, 1, 16, K, 1, 1);
  bb_mvout((uintptr_t)actual, 3, 4, 1);
  bb_fence();

  for (int col = 0; col < 16; ++col) {
    int expected = bias[col];
    for (int k = 0; k < K; ++k)
      expected += input[k] * weight[k * 16 + col];
    if (actual[col] != expected) {
      printf("smatmul_single_block FAIL col=%d expected=%d actual=%d\n", col,
             expected, actual[col]);
      return 1;
    }
  }
  printf("smatmul_single_block PASS\n");
  return 0;
}
