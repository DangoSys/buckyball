#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t expected_bits[16] = {
    0x3F800000, 0x40000000, 0x40400000, 0xBF800000, 0xC0000000, 0x00000000,
    0x40800000, 0x40A00000, 0x41200000, 0xC1200000, 0x40E00000, 0x42C80000,
    0xC2C80000, 0x41000000, 0x41800000, 0xC1000000,
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

#ifdef __cplusplus
extern "C"
#endif
    void check_result(float *allocated, float *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 4 || size1 != 4 || stride0 != 4 || stride1 != 1) {
    printf("FAILED: bank_ssa shape %dx%d stride %dx%d\n", (int)size0,
           (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  float *out = aligned + offset;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint32_t exp = expected_bits[i * 4 + j];
      uint32_t got = fp_bits(out[i * stride0 + j * stride1]);
      if (got != exp) {
        printf("FAILED: bank_ssa out[%d][%d] exp=0x%08X got=0x%08X\n", i, j,
               exp, got);
        fail();
      }
    }
  }
  printf("PASSED: int2fp bank_ssa 4x4 int32->fp32\n");
}
