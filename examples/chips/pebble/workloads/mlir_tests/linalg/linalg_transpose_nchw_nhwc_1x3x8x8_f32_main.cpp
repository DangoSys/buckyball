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
  if (s0 != 1 || s1 != 8 || s2 != 8 || s3 != 3)
    fail();
  float *out = aligned + offset;
  for (int h = 0; h < 8; ++h)
    for (int w = 0; w < 8; ++w)
      for (int c = 0; c < 3; ++c) {
        float got = out[h * st1 + w * st2 + c * st3];
        float exp = (float)(c + h + w);
        if (fabsf(got - exp) > 1e-5f)
          fail();
      }
  printf("PASSED: linalg.transpose nchw->nhwc 1x3x8x8 f32\n");
}
