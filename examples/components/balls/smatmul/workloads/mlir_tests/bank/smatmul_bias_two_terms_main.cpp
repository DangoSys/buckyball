#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != BANK_LINES || lanes != 4 || stride != 4 || lane_stride != 1)
    exit(1);
  for (int matrix_row = 0; matrix_row < BANK_LINES / 4; ++matrix_row)
    for (int column = 0; column < 16; ++column) {
      int bank_row = matrix_row * 4 + column / 4;
      int bank_lane = column % 4;
      int32_t expected = column - 8;
      for (int k = 0; k < 16; ++k)
        expected += (matrix_row % 16 - k) * ((k + column) + (2 * k - column));
      if (data[offset + bank_row * stride + bank_lane] != expected) {
        printf("smatmul bank mismatch row=%d column=%d\n", matrix_row, column);
        exit(1);
      }
    }
  printf("smatmul bank PASS rows=%d\n", BANK_LINES);
}
