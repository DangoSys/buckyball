#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

#ifdef __cplusplus
extern "C"
#endif
    void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 8 || size1 != 16 || stride0 != 16 || stride1 != 1) {
    printf(
        "FAILED: quant_im2col_k1_single_pixel shape %ldx%ld stride %ldx%ld\n",
        (long)size0, (long)size1, (long)stride0, (long)stride1);
    fail();
  }

  int8_t *out = aligned + offset;
  // The one window is line 0: the sixteen channels of the pixel.
  for (int channel = 0; channel < 16; ++channel) {
    int expected = channel < 4 ? channel + 1 : 0;
    int got = out[channel * stride1];
    if (got != expected) {
      printf("FAILED: quant_im2col_k1_single_pixel channel=%d exp=%d got=%d\n",
             channel, expected, got);
      fail();
    }
  }
  printf("PASSED: quant_im2col_k1_single_pixel\n");
}
