#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 16, N = 16, K = 16 };
enum {
  BIAS_BANK = VIRTUAL_BANK_NUM - 4,
  A_BANK = VIRTUAL_BANK_NUM - 3,
  B_BANK = VIRTUAL_BANK_NUM - 2,
  C_BANK = VIRTUAL_BANK_NUM - 1
};
static int8_t input[M * K] __attribute__((aligned(64)));
static int8_t weight[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t actual[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      input[row * K + col] = (3 * row + 5 * col) % 7 - 3;
      weight[row * N + col] = (4 * row + 3 * col) % 9 - 4;
    }
  for (int col = 0; col < N; ++col)
    bias[col] = 2 * col - 7;

  bb_mem_alloc(BIAS_BANK, 1, 1);
  bb_mem_alloc(A_BANK, 1, 1);
  bb_mem_alloc(B_BANK, 1, 1);
  bb_mem_alloc(C_BANK, 1, 1);
  bb_mvin((uintptr_t)bias, BIAS_BANK, 4, 1);
  bb_mvin((uintptr_t)input, A_BANK, M, 1);
  bb_mvin((uintptr_t)weight, B_BANK, K, 1);
  bb_smatmul_bias(BIAS_BANK, 0);
  bb_smatmul_os(A_BANK, B_BANK, C_BANK, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)actual, C_BANK, M * 4, 1);
  bb_fence();

  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int expected = bias[col];
      for (int k = 0; k < K; ++k)
        expected += input[row * K + k] * weight[k * N + col];
      if (actual[row * N + col] != expected) {
        printf("smatmul_nondefault_banks FAIL row=%d col=%d\n", row, col);
        return 1;
      }
    }
  printf("smatmul_nondefault_banks PASS\n");
  return 0;
}
