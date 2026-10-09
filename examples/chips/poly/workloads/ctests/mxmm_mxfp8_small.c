#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { M = 16, N = 16, K = 32 };
static uint8_t a[M * (K + K / 32)] __attribute__((aligned(64)));
static uint8_t b[N * (K + K / 32)] __attribute__((aligned(64)));
static float output[M * N] __attribute__((aligned(64)));

int main(void) {
  memset(a, 0x38, M * K);
  memset(b, 0x40, N * K);
  for (int row = 0; row < M; ++row)
    a[M * K + row] = row % 2 ? 128 : 127;
  for (int col = 0; col < N; ++col)
    b[N * K + col] = col % 2 ? 126 : 127;
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a, 3, sizeof(a) / 16, 1);
  bb_mvin((uintptr_t)b, 4, sizeof(b) / 16, 1);
  bb_mxmm_mxfp8(3, 4, 5, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)output, 5, sizeof(output) / 16, 1);
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      float expected =
          (float)K * (row % 2 ? 2.0f : 1.0f) * (col % 2 ? 1.0f : 2.0f);
      if (output[row * N + col] != expected) {
        printf("Mxmm MXFP8 small FAIL row=%d col=%d\n", row, col);
        return 1;
      }
    }
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_release(bank);
  puts("Mxmm MXFP8 small PASS");
  return 0;
}
