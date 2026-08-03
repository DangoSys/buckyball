#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const int32_t expected[16] = {
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 0, 100, -100, 7, 8, -8,
};

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
    void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 4 || size1 != 4 || stride0 != 4 || stride1 != 1) {
    printf("FAILED: bank_fp2int shape %dx%d stride %dx%d\n", (int)size0,
           (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  int32_t *out = aligned + offset;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int32_t exp = expected[i * 4 + j];
      int32_t got = out[i * stride0 + j * stride1];
      if (got != exp) {
        printf("FAILED: bank_fp2int out[%d][%d] exp=%d got=%d\n", i, j,
               (int)exp, (int)got);
        fail();
      }
    }
  }
  printf("PASSED: fp2int bank_fp2int 4x4 fp32->int32\n");
}
