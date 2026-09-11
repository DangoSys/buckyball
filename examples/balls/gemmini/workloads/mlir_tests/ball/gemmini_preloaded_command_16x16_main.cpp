#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 16 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < 16; ++row)
    for (int column = 0; column < 16; ++column) {
      int32_t expected = 0;
      for (int k = 0; k < 16; ++k)
        expected +=
            ((3 * row + 5 * k) % 7 - 3) * ((2 * k + 3 * column) % 5 - 2);
      if (data[offset + row * stride + column] != expected) {
        printf("gemmini ball mismatch row=%d column=%d\n", row, column);
        exit(1);
      }
    }
  printf("gemmini ball PASS\n");
}
