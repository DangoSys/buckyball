#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t expected_bits[16] = {
    0x3E000000, 0x3E800000, 0x3EC00000, 0xBE000000, 0xBE800000, 0x00000000,
    0x3F000000, 0x3F200000, 0x3FA00000, 0xBFA00000, 0x3F600000, 0x41480000,
    0xC1480000, 0x3F800000, 0x40000000, 0xBF800000,
};

static void fail(void) {
#ifdef BAREMETAL
  volatile uint32_t *sim_exit = (volatile uint32_t *)0x60000000;
  *sim_exit = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

static uint32_t fp_bits(float v) {
  uint32_t bits;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

extern "C" void prepare_scales() {
  static float da[4] __attribute__((aligned(64))) = {0.5f, 0.0f, 0.0f, 0.0f};
  static float dw[4] __attribute__((aligned(64))) = {0.25f, 0.0f, 0.0f, 0.0f};
  bb_mvin_mmio((uintptr_t)da, 0, 1, 4);
  bb_mvin_mmio((uintptr_t)dw, 16, 1, 4);
  bb_fence();
}

#ifdef __cplusplus
extern "C"
#endif
    void check_result(float *allocated, float *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 4 || size1 != 4 || stride0 != 4 || stride1 != 1) {
    printf("FAILED: ball_int2fp shape %dx%d stride %dx%d\n", (int)size0,
           (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  float *out = aligned + offset;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint32_t exp = expected_bits[i * 4 + j];
      uint32_t got = fp_bits(out[i * stride0 + j * stride1]);
      if (got != exp) {
        printf("FAILED: ball_int2fp out[%d][%d] exp=0x%08X got=0x%08X\n", i, j,
               exp, got);
        fail();
      }
    }
  }
  printf("PASSED: int2fp ball_int2fp 4x4 int32->fp32\n");
}
