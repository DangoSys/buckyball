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
                      int64_t size0, int64_t size1, int64_t size2,
                      int64_t size3, int64_t stride0, int64_t stride1,
                      int64_t stride2, int64_t stride3) {
  (void)allocated;
  (void)stride0;
  const int64_t sizes[4] = {1, 3, 3, 4};
  if (size0 != sizes[0] || size1 != sizes[1] || size2 != sizes[2] ||
      size3 != sizes[3]) {
    printf("FAILED: linalg conv2d_nhwc_hwcf unexpected memref shape\n");
    fail();
  }
  const float *output = aligned + offset;
  for (int oh = 0; oh < 3; ++oh)
    for (int ow = 0; ow < 3; ++ow)
      for (int f = 0; f < 4; ++f) {
        int sum = 0;
        for (int kh = 0; kh < 3; ++kh)
          for (int kw = 0; kw < 3; ++kw)
            for (int c = 0; c < 3; ++c) {
              int x = (3 * (oh + kh) + 5 * (ow + kw) + 7 * c) % 9 - 4;
              int w = (2 * f + 3 * kh + 5 * kw + c) % 7 - 3;
              sum += x * w;
            }
        float expected = 1.0f + 0.125f * (float)sum;
        float got = output[oh * stride1 + ow * stride2 + f * stride3];
        if (got != expected) {
          printf(
              "FAILED: linalg conv2d_nhwc_hwcf oh=%d ow=%d f=%d expected %d/8, "
              "got %d/8\n",
              oh, ow, f, (int)(expected * 8), (int)(got * 8));
          fail();
        }
      }
  printf("PASSED: linalg conv2d_nhwc_hwcf\n");
}
