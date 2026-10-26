#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

/* M=8, N=16, K=48 in one SMATMUL_OS: A and B have three groups, group g holds
 * reduction block g (k = 16 per group). Row-major A[8][48] lands in a
 * three-group bank as group g line r = A[r][16g..16g+15]. */
enum { M = 8, N = 16, K = 48 };
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t actual[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      a[row * K + k] = (row * 5 + k * 3) % 13 - 6;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      b[k * N + col] = (k * 7 + col) % 11 - 5;
  for (int col = 0; col < N; ++col)
    bias[col] = col * 3 - 20;

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 3);
  bb_mem_alloc(2, 1, 3);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)a, 1, M, 1);
  bb_mvin_group((uintptr_t)b, 2, 0, 16, 1);
  bb_mvin_group((uintptr_t)(b + 16 * N), 2, 1, 16, 1);
  bb_mvin_group((uintptr_t)(b + 32 * N), 2, 2, 16, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, M, N, 16, 1, 1, 0);
  bb_mvout((uintptr_t)actual, 3, M * 4, 1);

  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int expected = bias[col];
      for (int k = 0; k < K; ++k)
        expected += a[row * K + k] * b[k * N + col];
      if (actual[row * N + col] != expected) {
        printf("smatmul_k_groups FAIL row=%d col=%d expected=%d actual=%d\n",
               row, col, expected, actual[row * N + col]);
        return 1;
      }
    }
  printf("smatmul_k_groups PASS\n");
  return 0;
}
