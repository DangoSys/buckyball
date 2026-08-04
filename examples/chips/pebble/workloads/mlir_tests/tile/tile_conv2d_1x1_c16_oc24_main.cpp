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
                      int64_t n, int64_t h, int64_t w, int64_t c, int64_t sn,
                      int64_t sh, int64_t sw, int64_t sc) {
  (void)allocated;
  if (n != 1 || h != 1 || w != 1 || c != 24 || sn != 24 || sh != 24 ||
      sw != 24 || sc != 1)
    fail();
  // OC tiled by 16: each OC sums 16 ones -> 16
  for (int i = 0; i < 24; ++i) {
    if (aligned[offset + i] != 16.0f)
      fail();
  }
  printf("PASSED: tile.tile_conv2d 1x1 c16 oc24\n");
}
