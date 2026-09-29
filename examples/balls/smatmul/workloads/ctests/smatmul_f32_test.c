#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { N = 16, K = 8 };
static float a[16 * K] __attribute__((aligned(64)));
static float b[N * K] __attribute__((aligned(64)));
static float actual[16 * N + 4] __attribute__((aligned(64)));

int main(void) {
  for (int m = 1; m <= 16; m += 15) {
    for (int row = 0; row < m; ++row)
      for (int k = 0; k < K; ++k)
        a[row * K + k] = ((row + 2 * k) % 9 - 4) * 0.25f;
    for (int col = 0; col < N; ++col)
      for (int k = 0; k < K; ++k)
        b[col * K + k] = ((3 * k + col) % 11 - 5) * 0.125f;
    for (int bank = 0; bank < 3; ++bank)
      bb_mem_alloc(bank, 1, 1);
    bb_mset_clear(2, 1, 1);
    bb_mvin((uintptr_t)a, 0, m * K / 4, 1);
    bb_mvin((uintptr_t)b, 1, N * K / 4, 1);
    bb_smatmul_f32(0, 1, 2, m, N, K, 1, 0, 1);
    bb_smatmul_f32(0, 1, 2, m, N, K, 0, 1, 1);
    bb_mvout((uintptr_t)actual, 2, m * N / 4 + 1, 1);
    bb_fence();
    for (int i = 0; i < 4; ++i)
      if (actual[i] != 0)
        return 1;
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < N; ++col) {
        float expected = 0;
        for (int pass = 0; pass < 2; ++pass)
          for (int k = 0; k < K; ++k)
            expected += a[row * K + k] * b[col * K + k];
        if (actual[4 + row * N + col] != expected) {
          printf("smatmul_f32 FAIL m=%d row=%d col=%d\n", m, row, col);
          return 2;
        }
      }
    for (int bank = 0; bank < 3; ++bank)
      bb_mem_release(bank);
  }
  puts("smatmul_f32 PASS");
  return 0;
}
