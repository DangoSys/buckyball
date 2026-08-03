#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                      int64_t n, int64_t h, int64_t w, int64_t c, int64_t sn,
                      int64_t sh, int64_t sw, int64_t sc) {
  (void)allocated;
  if (n != 1 || h != 4 || w != 4 || c != 1 || sn != 16 || sh != 4 || sw != 1 ||
      sc != 1)
    fail();
  for (int i = 0; i < 16; ++i) {
    float value = aligned[offset + i];
    if (value != 9.0f)
      fail();
  }
  printf("PASSED: linalg.conv_2d_nhwc_hwcf 6x6 k3\n");
}
