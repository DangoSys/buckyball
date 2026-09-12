#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 16, N = 16, K = 32 };
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t packed_a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t actual[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      a[row * K + k] = (row + 2 * k) % 9 - 4;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      b[k * N + col] = (3 * k + col) % 11 - 5;
  for (int col = 0; col < N; ++col)
    bias[col] = 2 * col - 9;
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      packed_a[((k / 16) * 16 + row) * 16 + k % 16] = a[row * K + k];

  for (int bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)packed_a, 1, M * K / 16, 1);
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
        printf("smatmul_16x16x32 FAIL row=%d col=%d expected=%d actual=%d\n",
               row, col, expected, actual[index]);
        return 1;
      }
    }
  printf("smatmul_16x16x32 PASS\n");
  return 0;
}
