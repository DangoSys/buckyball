#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxmm.h>
#include <stdio.h>

enum { M = 16, N = 16, K = 32 };
static float a[M * K] __attribute__((aligned(64)));
static float b[N * K] __attribute__((aligned(64)));
static float output[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      a[row * K + k] = (float)(row + 1);
  for (int col = 0; col < N; ++col)
    for (int k = 0; k < K; ++k)
      b[col * K + k] = (float)(col + 1);
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a, 3, sizeof(a) / 16, 1);
  bb_mvin((uintptr_t)b, 4, sizeof(b) / 16, 1);
  bb_mxmm_f32(3, 4, 5, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)output, 5, sizeof(output) / 16, 1);
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col)
      if (output[row * N + col] != (float)(K * (row + 1) * (col + 1))) {
        printf("Mxmm FP32 small FAIL row=%d col=%d\n", row, col);
        return 1;
      }
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_release(bank);
  puts("Mxmm FP32 small PASS");
  return 0;
}
