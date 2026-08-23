#include <math.h>
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

#ifdef __cplusplus
extern "C"
#endif
    void check_result(float *allocated, float *aligned, int64_t offset,
                      int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                      int64_t st0, int64_t st1, int64_t st2, int64_t st3) {
  (void)allocated;
  if (s0 != 4 || s1 != 3 || s2 != 3 || s3 != 8)
    fail();
  float *out = aligned + offset;
  for (int o = 0; o < 4; ++o)
    for (int h = 0; h < 3; ++h)
      for (int w = 0; w < 3; ++w)
        for (int i = 0; i < 8; ++i) {
          float got = out[o * st0 + h * st1 + w * st2 + i * st3];
          float exp = (float)(o + i + h + w);
          if (fabsf(got - exp) > 1e-5f)
            fail();
        }
  printf("PASSED: linalg.transpose oihw->ohwi 4x8x3x3 f32\n");
}
