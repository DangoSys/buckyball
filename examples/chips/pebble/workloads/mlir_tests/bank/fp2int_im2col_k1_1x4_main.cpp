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
  if (size0 != 16 || size1 != 16 || stride0 != 16 || stride1 != 1) {
    printf("FAILED: bank_fp2int_im2col_k1_1x4 shape %ldx%ld stride %ldx%ld\n",
           (long)size0, (long)size1, (long)stride0, (long)stride1);
    fail();
  }

  int8_t *out = aligned + offset;
  int32_t got = out[0 * stride0 + 0 * stride1];
  if (got != 32) {
    printf("FAILED: bank_fp2int_im2col_k1_1x4 out[0][0] exp=32 got=%d\n",
           (int)got);
    fail();
  }
  printf("PASSED: bank_fp2int_im2col_k1_1x4 (MobileNet tile0 smoke)\n");
}
