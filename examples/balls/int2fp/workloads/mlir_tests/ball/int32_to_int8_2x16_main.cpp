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
  static const int8_t expected[32] = {
      -128, -128, -128, -2, -2, 0,  0, 0, 2, 2, 64, 126, 127, 127, 127, 1,
      -128, -128, -128, -4, -4, -2, 1, 2, 3, 4, 62, 126, 127, 127, 127, 127,
  };
  (void)allocated;
  if (size0 != 2 || size1 != 16 || stride0 != 16 || stride1 != 1) {
    fail();
  }
  int8_t *out = aligned + offset;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 16; ++j) {
      if (out[i * stride0 + j * stride1] != expected[i * 16 + j]) {
        fail();
      }
    }
  }
  printf("PASSED: int2fp ball_int32_to_int8 2x16\n");
}
