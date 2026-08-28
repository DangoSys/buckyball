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
                      int64_t s0, int64_t s1, int64_t s2, int64_t st0,
                      int64_t st1, int64_t st2) {
  (void)allocated;
  if (s0 != 1 || s1 != 100 || s2 != 84)
    fail();
  float *out = aligned + offset;
  for (int b = 0; b < 100; ++b)
    for (int a = 0; a < 84; ++a) {
      float got = out[b * st1 + a * st2];
      float exp = (float)(a + b);
      if (fabsf(got - exp) > 1e-5f)
        fail();
    }
  printf("PASSED: linalg.transpose detect 1x84x100 f32\n");
}
