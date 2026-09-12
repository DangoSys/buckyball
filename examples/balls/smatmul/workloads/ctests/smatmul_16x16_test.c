#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 16, N = 16, K = 16 };
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t actual[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      a[row * K + k] = (row * 3 + k * 5) % 13 - 6;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      b[k * N + col] = (k * 7 + col * 2) % 11 - 5;
  for (int col = 0; col < N; ++col)
    bias[col] = col - 8;

  for (int bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)a, 1, M, 1);
  bb_mvin((uintptr_t)b, 2, K, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)actual, 3, M * 4, 1);
  bb_fence();

  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int expected = bias[col];
      for (int k = 0; k < K; ++k)
        expected += a[row * K + k] * b[k * N + col];
      int index = (row * 4 + col / 4) * 4 + col % 4;
      if (actual[index] != expected) {
        printf("smatmul_16x16 FAIL row=%d col=%d expected=%d actual=%d\n", row,
               col, expected, actual[index]);
        return 1;
      }
    }
  printf("smatmul_16x16 PASS\n");
  return 0;
}
