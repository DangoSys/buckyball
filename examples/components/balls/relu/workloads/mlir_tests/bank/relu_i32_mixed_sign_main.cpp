#include <params.h>
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
    void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != BANK_LINES || size1 != BANK_WIDTH / 32 ||
      stride0 != BANK_WIDTH / 32 || stride1 != 1) {
    printf("FAILED: relu bank shape %dx%d stride %dx%d\n", (int)size0,
           (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  int32_t *output = aligned + offset;
  for (int line = 0; line < BANK_LINES; ++line) {
    for (int lane = 0; lane < BANK_WIDTH / 32; ++lane) {
      int32_t input = (line * (BANK_WIDTH / 32) + lane) % 17 - 8;
      int32_t expected = input < 0 ? 0 : input;
      int32_t actual = output[line * stride0 + lane * stride1];
      if (actual != expected) {
        printf("FAILED: relu bank[%d][%d] expected %d got %d\n", line, lane,
               expected, actual);
        fail();
      }
    }
  }
  printf("relu i32 mixed-sign PASS rows=%d\n", BANK_LINES);
}
