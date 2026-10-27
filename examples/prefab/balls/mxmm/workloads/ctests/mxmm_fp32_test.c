#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { K = 64, MAX_M = 32, MAX_N = 48 };

static uint32_t a[MAX_M * K] __attribute__((aligned(64)));
static uint32_t b[MAX_N * K] __attribute__((aligned(64)));
static uint32_t actual[4 + MAX_M * MAX_N] __attribute__((aligned(64)));

int main(void) {
  for (int m = 1; m <= MAX_M; m = m == 1 ? 16 : m + 16)
    for (int n = 16; n <= MAX_N; n += 32)
      for (int fused = 0; fused <= 1; ++fused) {
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        for (int row = 0; row < m; ++row) {
          a[row * K] = 0xbf800000;
          a[row * K + 1] = 0x3f800001;
        }
        for (int col = 0; col < n; ++col) {
          b[col * K] = 0x3f800000;
          b[col * K + 1] = 0x3f7ffffe;
          b[col * K + 2] = b[col * K + 3] = 0x3f800000;
        }
        for (int i = 0; i < 4 + m * n; ++i)
          actual[i] = 0x5a5a5a5a;
        // Publish CPU input/output initialization once before this DMA chain.
        for (int bank = 3; bank <= 5; ++bank)
          bb_mem_alloc(bank, 1, 1);
        bb_mvin((uintptr_t)a, 3, m * K / 4, 1);
        bb_mvin((uintptr_t)b, 4, n * K / 4, 1);
        bb_mvin((uintptr_t)actual, 5, 1 + m * n / 4, 1);
        if (fused)
          bb_mxmm_fma32(3, 4, 5, m, n, K, 1, 0, 1);
        else
          bb_mxmm_f32(3, 4, 5, m, n, K, 1, 0, 1);
        bb_mvout((uintptr_t)actual, 5, 1 + m * n / 4, 1);
        for (int i = 0; i < 4 + m * n; ++i)
          if (actual[i] != 0x5a5a5a5a)
            return 1;
        for (int i = 0; i < m * K; ++i)
          a[i] = 0;
        bb_mvin((uintptr_t)a, 3, m * K / 4, 1);
        if (fused)
          bb_mxmm_fma32(3, 4, 5, m, n, K, 0, 1, 1);
        else
          bb_mxmm_f32(3, 4, 5, m, n, K, 0, 1, 1);
        bb_mvout((uintptr_t)actual, 5, 1 + m * n / 4, 1);
        for (int i = 0; i < 4; ++i)
          if (actual[i] != 0x5a5a5a5a)
            return 2;
        for (int i = 4; i < 4 + m * n; ++i)
          if (actual[i] != (fused ? 0xa8800000u : 0)) {
            printf("matmul FP32 FAIL m=%d fused=%d i=%d bits=%08x\n", m, fused,
                   i, actual[i]);
            return 3;
          }
        for (int bank = 3; bank <= 5; ++bank)
          bb_mem_release(bank);
      }
  puts("matmul FP32 fused/unfused PASS");
  return 0;
}
