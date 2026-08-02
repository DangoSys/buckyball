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
    printf("FAILED: ball_op shape %dx%d stride %dx%d\n", (int)size0, (int)size1,
           (int)stride0, (int)stride1);
    fail();
  }

  int8_t *out = aligned + offset;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int8_t exp = (int8_t)(j * 16 + i);
      int8_t got = out[i * stride0 + j * stride1];
      if (got != exp) {
        printf("FAILED: ball_op out[%d][%d] exp=%d got=%d\n", i, j, (int)exp,
               (int)got);
        fail();
      }
    }
  }
  printf("PASSED: transpose ball_op 16x16 i8\n");
}
