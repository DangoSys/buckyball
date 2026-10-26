#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { K = 256, MAX_M = 32, MAX_N = 48 };

static uint8_t a[MAX_M * (K + K / 32)] __attribute__((aligned(64)));
static uint8_t b[MAX_N * (K + K / 32)] __attribute__((aligned(64)));
static uint32_t actual[4 + MAX_M * MAX_N] __attribute__((aligned(64)));

int main(void) {
  for (int m = 1; m <= MAX_M; m = m == 1 ? 16 : m + 16)
    for (int n = 16; n <= MAX_N; n += 32) {
      memset(a, 0, sizeof(a));
      for (int row = 0; row < m; ++row) {
        memset(a + row * K, 0x38, K);
        memset(a + m * K + row * (K / 32), row % 2 ? 128 : 127, K / 32);
      }
      for (int col = 0; col < n; ++col) {
        memset(b + col * K, 0x40, K);
        memset(b + n * K + col * (K / 32), col % 2 ? 126 : 127, K / 32);
      }
      for (int i = 0; i < 4 + m * n; ++i)
        actual[i] = 0x5a5a5a5a;
      // Publish CPU input/output initialization once before this DMA chain.
      for (int bank = 3; bank <= 5; ++bank)
        bb_mem_alloc(bank, 1, 1);
      bb_mvin((uintptr_t)a, 3, (m * (K + K / 32) + 15) / 16, 1);
      bb_mvin((uintptr_t)b, 4, n * (K + K / 32) / 16, 1);
      bb_mvin((uintptr_t)actual, 5, 1 + m * n / 4, 1);
      bb_mxmm_mxfp8(3, 4, 5, m, n, K, 1, 0, 1);
      bb_mvout((uintptr_t)actual, 5, 1 + m * n / 4, 1);
      for (int i = 0; i < 4 + m * n; ++i)
        if (actual[i] != 0x5a5a5a5a)
          return 1;
      for (int row = 0; row < m; ++row)
        memset(a + row * K, 0xb8, K);
      for (int col = 0; col < n; ++col)
        memset(b + col * K, 0x38, K);
      bb_mvin((uintptr_t)a, 3, (m * (K + K / 32) + 15) / 16, 1);
      bb_mvin((uintptr_t)b, 4, n * (K + K / 32) / 16, 1);
      bb_mxmm_mxfp8(3, 4, 5, m, n, K, 0, 1, 1);
      bb_mvout((uintptr_t)actual, 5, 1 + m * n / 4, 1);
      for (int i = 0; i < 4; ++i)
        if (actual[i] != 0x5a5a5a5a)
          return 2;
      for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
          union {
            float value;
            uint32_t bits;
          } expected;
          expected.value =
              (float)K * (row % 2 ? 2.0f : 1.0f) * (col % 2 ? 0.5f : 1.0f);
          if (actual[4 + row * n + col] != expected.bits)
            return 3;
        }
      memset(a, 0, sizeof(a));
      memset(b, 0, sizeof(b));
      memset(a, 0x81, m * K);
      memset(b, 0x01, n * K);
      bb_mvin((uintptr_t)a, 3, (m * (K + K / 32) + 15) / 16, 1);
      bb_mvin((uintptr_t)b, 4, n * (K + K / 32) / 16, 1);
      bb_mxmm_mxfp8(3, 4, 5, m, n, K, 1, 1, 1);
      bb_mvout((uintptr_t)actual, 5, 1 + m * n / 4, 1);
      for (int i = 4; i < 4 + m * n; ++i)
        if (actual[i] != 0x80000000u)
          return 4;
      for (int bank = 3; bank <= 5; ++bank)
        bb_mem_release(bank);
    }
  puts("matmul MXFP8 chain/subnormal PASS");
  return 0;
}
