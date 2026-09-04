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

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  (void)allocated;
  if (rows != 12 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  int8_t *output = aligned + offset;
  for (int row = 0; row < 12; ++row) {
    for (int channel = 0; channel < 16; ++channel) {
      int position = row == 3   ? 0
                     : row == 4 ? 1
                     : row == 7 ? 2
                     : row == 8 ? 3
                                : -1;
      int expected = position < 0 ? 0 : position * 20 + channel - 8;
      if (expected < 0)
        expected = 0;
      int actual = output[row * row_stride + channel * column_stride];
      if (actual != expected) {
        printf(
            "FAILED: quant_i32_to_i8 row=%d channel=%d expected=%d actual=%d\n",
            row, channel, expected, actual);
        fail();
      }
    }
  }
  printf("PASSED: quant_i32_to_i8 relu and strided output\n");
}
