#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 16 || lanes != 4 || stride != 4 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int lane = 0; lane < lanes; ++lane) {
      int input = row * lanes + lane - 16;
      int expected = input < 0 ? 0 : input;
      if (data[offset + row * stride + lane] != expected)
        exit(1);
    }
  printf("relu i32 16-row ball PASS\n");
}
