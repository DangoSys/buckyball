#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 1, N = 16, K = 64 };
static int8_t input[K] __attribute__((aligned(64)));
static int8_t weight[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t actual[N] __attribute__((aligned(64)));

int main(void) {
  for (int k = 0; k < K; ++k)
    input[k] = k % 7 - 3;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      weight[k * N + col] = (k + col) % 11 - 5;
  for (int col = 0; col < N; ++col)
    bias[col] = 3 * col - 8;

  for (int bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)input, 1, K / 16, 1);
  bb_mvin((uintptr_t)weight, 2, K, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)actual, 3, 4, 1);
  bb_fence();

  for (int col = 0; col < N; ++col) {
    int expected = bias[col];
    for (int k = 0; k < K; ++k)
      expected += input[k] * weight[k * N + col];
    if (actual[col] != expected) {
      printf("smatmul_1x16x64 FAIL col=%d expected=%d actual=%d\n", col,
             expected, actual[col]);
      return 1;
    }
  }
  printf("smatmul_1x16x64 PASS\n");
  return 0;
}
