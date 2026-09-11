#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != BANK_LINES || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < BANK_LINES; ++row)
    for (int column = 0; column < 16; ++column) {
      int32_t expected = 0;
      for (int k = 0; k < 16; ++k)
        expected +=
            ((3 * (row % 16) + 5 * k) % 7 - 3) * ((2 * k + 3 * column) % 5 - 2);
      if (data[offset + row * stride + column] != expected) {
        printf("gemmini bank mismatch row=%d column=%d expected=%d actual=%d\n",
               row, column, expected, data[offset + row * stride + column]);
        exit(1);
      }
    }
  printf("gemmini compute-preloaded bank tiles PASS rows=%d\n", BANK_LINES);
}
