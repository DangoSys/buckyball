#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

/* M=128, N=16, K=32 with output_base 32: C occupies lines 32..543, so its rows
 * continue from group 0 into group 1 (line L is group L /
 * THREAD_MEM_BANK_ENTRIES). A and B have two reduction groups. */
enum { M = 128, N = 16, K = 32, BASE = 32 };
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t c[2][THREAD_MEM_BANK_ENTRIES * 4] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      a[row * K + k] = (row * 5 + k * 3) % 13 - 6;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      b[k * N + col] = (k * 7 + col) % 11 - 5;
  for (int col = 0; col < N; ++col)
    bias[col] = col - 8;

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 2);
  bb_mem_alloc(2, 1, 2);
  bb_mem_alloc(3, 1, 2);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)a, 1, M, 1);
  bb_mvin_group((uintptr_t)b, 2, 0, 16, 1);
  bb_mvin_group((uintptr_t)(b + 16 * N), 2, 1, 16, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, M, N, 16, 1, 1, BASE);
  bb_mvout_group((uintptr_t)c[0], 3, 0, THREAD_MEM_BANK_ENTRIES, 1);
  bb_mvout_group((uintptr_t)c[1], 3, 1, THREAD_MEM_BANK_ENTRIES, 1);

  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int expected = bias[col];
      for (int k = 0; k < K; ++k)
        expected += a[row * K + k] * b[k * N + col];
      int line = BASE + row * 4 + col / 4;
      int actual = c[line / THREAD_MEM_BANK_ENTRIES]
                    [line % THREAD_MEM_BANK_ENTRIES * 4 + col % 4];
      if (actual != expected) {
        printf("smatmul_c_groups FAIL row=%d col=%d expected=%d actual=%d\n",
               row, col, expected, actual);
        return 1;
      }
    }
  printf("smatmul_c_groups PASS\n");
  return 0;
}
