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

extern "C" void check_result(int32_t *allocated, int32_t *aligned,
                             int64_t offset, int64_t rows, int64_t lanes,
                             int64_t row_stride, int64_t lane_stride) {
  (void)allocated;
  if (rows != 64 || lanes != 4 || row_stride != 4 || lane_stride != 1)
    fail();

  int32_t *output = aligned + offset;
  for (int row = 0; row < 16; ++row) {
    for (int column = 0; column < 16; ++column) {
      int bank_row = row * 4 + column / 4;
      int bank_lane = column % 4;
      int32_t actual = output[bank_row * row_stride + bank_lane * lane_stride];
      int32_t expected = 2 * row + column - 8;
      if (actual != expected) {
        printf("FAILED: smatmul bias row=%d column=%d expected=%d actual=%d\n",
               row, column, expected, actual);
        fail();
      }
    }
  }
  printf("PASSED: smatmul bias and two-step accumulation\n");
}
