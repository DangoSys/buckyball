#include <bbsw/test.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif
    void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t size2,
                      int64_t size3, int64_t stride0, int64_t stride1,
                      int64_t stride2, int64_t stride3) {
  (void)allocated;

  if (size0 != 1 || size1 != 6 || size2 != 6 || size3 != 32 ||
      stride0 != 1152 || stride1 != 192 || stride2 != 32 || stride3 != 1) {
    printf("FAILED: linalg conv2d NHWC unexpected memref shape "
           "(size=%dx%dx%dx%d stride=%dx%dx%dx%d)\n",
           (int)size0, (int)size1, (int)size2, (int)size3, (int)stride0,
           (int)stride1, (int)stride2, (int)stride3);
    bb_test_fail();
  }

  int32_t *output = aligned + offset;
  const int32_t expected = 0x43100000;
  for (int i = 0; i < 1; ++i) {
    for (int h = 0; h < 6; ++h) {
      for (int w = 0; w < 6; ++w) {
        for (int c = 0; c < 32; ++c) {
          int64_t idx = i * stride0 + h * stride1 + w * stride2 + c * stride3;
          int32_t got = output[idx];
          if (got != expected) {
            printf("FAILED: linalg conv2d NHWC output[%d][%d][%d][%d] "
                   "(expected 0x%08x, got 0x%08x)\n",
                   i, h, w, c, expected, got);
            bb_test_fail();
          }
        }
      }
    }
  }

  printf("PASSED: linalg conv2d NHWC\n");
  bb_test_pass();
}
