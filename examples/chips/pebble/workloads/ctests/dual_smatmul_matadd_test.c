#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 16, N = 16, K = 16, C_ROWS = M * 4 };
static int8_t a0[M * K] __attribute__((aligned(64)));
static int8_t a1[M * K] __attribute__((aligned(64)));
static int8_t b0[K * N] __attribute__((aligned(64)));
static int8_t b1[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static int32_t c0[C_ROWS * 4] __attribute__((aligned(64)));
static int32_t c1[C_ROWS * 4] __attribute__((aligned(64)));
static int32_t out[C_ROWS * 4] __attribute__((aligned(64)));

static void run(const int8_t *a, const int8_t *b, int32_t *c, int out_bank) {
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_alloc(out_bank, 1, 1);
  bb_mvin((uintptr_t)bias, 0, 4, 1);
  bb_mvin((uintptr_t)a, 1, M, 1);
  bb_mvin((uintptr_t)b, 2, K, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, out_bank, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)c, out_bank, C_ROWS, 1);
  bb_fence();
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_release(bank);
  bb_mem_release(out_bank);
}

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < K; ++col) {
      a0[row * K + col] = (3 * row + 5 * col) % 7 - 3;
      a1[row * K + col] = (2 * row + col) % 5 - 2;
    }
  for (int row = 0; row < K; ++row)
    for (int col = 0; col < N; ++col) {
      b0[row * N + col] = (4 * row + 3 * col) % 9 - 4;
      b1[row * N + col] = (5 * row + 2 * col) % 11 - 5;
    }
  for (int col = 0; col < N; ++col)
    bias[col] = col - 8;
  run(a0, b0, c0, 3);
  run(a1, b1, c1, 4);
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)c0, 0, C_ROWS, 1);
  bb_mvin((uintptr_t)c1, 1, C_ROWS, 1);
  bb_matadd(0, 1, 2, C_ROWS);
  bb_mvout((uintptr_t)out, 2, C_ROWS, 1);
  bb_fence();
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int index = (row * 4 + col / 4) * 4 + col % 4;
      int expected = 2 * bias[col];
      for (int k = 0; k < K; ++k)
        expected += a0[row * K + k] * b0[k * N + col] +
                    a1[row * K + k] * b1[k * N + col];
      if (out[index] != expected) {
        printf("dual smatmul matadd mismatch row=%d col=%d\n", row, col);
        return 1;
      }
    }
  printf("dual smatmul matadd PASS\n");
  return 0;
}
