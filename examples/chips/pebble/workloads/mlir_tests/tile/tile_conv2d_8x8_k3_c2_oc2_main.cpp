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
  if (n != 1 || h != 6 || w != 6 || c != 2 || sn != 72 || sh != 12 || sw != 2 ||
      sc != 1)
    fail();
  // quant scale=1: each window sums 3*3*2 ones -> 18
  for (int i = 0; i < 6 * 6 * 2; ++i) {
    if (aligned[offset + i] != 18.0f)
      fail();
  }
  printf("PASSED: tile.tile_conv2d 8x8 k3 c2 oc2\n");
}
