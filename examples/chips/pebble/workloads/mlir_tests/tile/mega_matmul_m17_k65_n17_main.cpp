#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail() {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

extern "C" void check_result(float *allocated, float *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  (void)allocated;
  if (rows != 17 || columns != 17 || row_stride != 17 || column_stride != 1)
    fail();
  for (int row = 0; row < 17; ++row)
    for (int column = 0; column < 17; ++column) {
      int sum = column - 8;
      for (int k = 0; k < 65; ++k)
        sum += ((row + k) % 7 - 3) * ((2 * k + 3 * column) % 5 - 2);
      float actual = data[offset + row * row_stride + column * column_stride];
      if (actual != (float)sum * 0.5f) {
        printf(
            "FAILED: segmented matmul row=%d column=%d expected=%f actual=%f\n",
            row, column, (float)sum * 0.5f, actual);
        fail();
      }
    }
  printf("PASSED: segmented matmul M17 K65 N17\n");
}
