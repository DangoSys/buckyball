#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 9 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int lane = 0; lane < lanes; ++lane) {
        int expected = (y * 2 + 1) * 6 + x * 2 + 1;
        if (data[offset + (y * 3 + x) * stride + lane] != expected) {
          printf("maxpool ball mismatch y=%d x=%d lane=%d\n", y, x, lane);
          exit(1);
        }
      }
  printf("maxpool ball PASS\n");
}
