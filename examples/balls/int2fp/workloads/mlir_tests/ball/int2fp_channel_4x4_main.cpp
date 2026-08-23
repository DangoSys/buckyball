#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

static const float expected[16] = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  7.0f,  8.0f,
    0.5f, 1.0f, 1.5f, 2.0f, 9.0f, 10.0f, 11.0f, 12.0f,
};

extern "C" void prepare_scales() {
  static float da[4] __attribute__((aligned(64))) = {0.5f};
  static float dw[16] __attribute__((aligned(64))) = {
      0.25f,  0.5f,  0.75f,  1.0f, 1.25f, 1.5f, 1.75f, 2.0f,
      0.125f, 0.25f, 0.375f, 0.5f, 2.25f, 2.5f, 2.75f, 3.0f,
  };
  bb_mvin_mmio((uintptr_t)da, 0, 1, 4);
  bb_mvin_mmio((uintptr_t)dw, 16, 4, 16);
  bb_fence();
}

extern "C" void check_result(float *, float *out, int64_t offset, int64_t rows,
                             int64_t cols, int64_t stride0, int64_t stride1) {
  if (rows != 1 || cols != 16 || stride0 != 16 || stride1 != 1)
    fail();
  for (int i = 0; i < 16; ++i) {
    if (out[offset + i] != expected[i])
      fail();
  }
  printf("PASSED: int2fp_channel_4x4 ball\n");
}
