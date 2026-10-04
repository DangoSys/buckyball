#include "mxmm_native_steps.h"
#include <bbhw/isa/isa.h>
#include <buckyball.h>
#include <dma.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum class MxmmFormat { Mxfp8, Fma32, F32 };

template <MxmmFormat Format> static int run_mxmm_native_test() {
  constexpr int K = Format == MxmmFormat::Mxfp8 ? 256 : 64;
  constexpr int bytes = BANK_LINES * BANK_WIDTH / 8;
  alignas(bytes) uint8_t a[bytes]{}, b[bytes]{};
  alignas(bytes) uint32_t result[bytes / 4];
  for (int mindex = 0; mindex < 3; mindex++)
    for (int nindex = 0; nindex < 2; nindex++) {
      int m = mindex == 0 ? 1 : mindex * 16, n = nindex == 0 ? 16 : 48;
      memset(a, 0, sizeof(a));
      memset(b, 0, sizeof(b));
      if (Format == MxmmFormat::Mxfp8) {
        memset(a, 0x38, m * K);
        memset(a + m * K, 127, m * K / 32);
        memset(b, 0x40, n * K);
        memset(b + n * K, 127, n * K / 32);
      } else {
        auto *x = (uint32_t *)a;
        auto *y = (uint32_t *)b;
        for (int r = 0; r < m; r++) {
          x[r * K] = 0xbf800000;
          x[r * K + 1] = 0x3f800001;
        }
        for (int c = 0; c < n; c++) {
          y[c * K] = 0x3f800000;
          y[c * K + 1] = 0x3f7ffffe;
        }
      }
      for (auto &v : result)
        v = 0x5a5a5a5a;
      for (int bank = 3; bank <= 5; bank++)
        bb_mem_alloc(bank, 1, 1);
      int ar = Format == MxmmFormat::Mxfp8 ? (m * K * 33 / 32 + 15) / 16
                                           : m * K / 4,
          br = Format == MxmmFormat::Mxfp8 ? n * K * 33 / 32 / 16 : n * K / 4;
      bb_mvin((uintptr_t)a, 3, ar, 1);
      bb_mvin((uintptr_t)b, 4, br, 1);
      bb_mvin((uintptr_t)result, 5, 1 + m * n / 4, 1);
      steps[mindex][nindex][0]();
      bb_mvout((uintptr_t)result, 5, 1 + m * n / 4, 1);
      bb_fence();
      for (int i = 0; i < 4 + m * n; i++)
        if (result[i] != 0x5a5a5a5a)
          return 1;
      if (Format == MxmmFormat::Mxfp8) {
        memset(a, 0xb8, m * K);
        memset(b, 0x38, n * K);
      } else
        memset(a, 0, m * K * 4);
      bb_mvin((uintptr_t)a, 3, ar, 1);
      bb_mvin((uintptr_t)b, 4, br, 1);
      steps[mindex][nindex][1]();
      bb_mvout((uintptr_t)result, 5, 1 + m * n / 4, 1);
      bb_fence();
      uint32_t expected = Format == MxmmFormat::Mxfp8   ? 0x43800000
                          : Format == MxmmFormat::Fma32 ? 0xa8800000
                                                        : 0;
      for (int i = 0; i < 4; i++)
        if (result[i] != 0x5a5a5a5a)
          return 2;
      for (int i = 4; i < 4 + m * n; i++)
        if (result[i] != expected)
          return 3;
      if (Format == MxmmFormat::Mxfp8) {
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        memset(a, 0x81, m * K);
        memset(b, 1, n * K);
        bb_mvin((uintptr_t)a, 3, ar, 1);
        bb_mvin((uintptr_t)b, 4, br, 1);
        steps[mindex][nindex][2]();
        bb_mvout((uintptr_t)result, 5, 1 + m * n / 4, 1);
        bb_fence();
        for (int i = 4; i < 4 + m * n; i++)
          if (result[i] != 0x80000000)
            return 4;
      }
      for (int bank = 3; bank <= 5; bank++)
        bb_mem_release(bank);
    }
  puts("Mxmm MLIR mode/shape/chain PASS");
  return 0;
}
