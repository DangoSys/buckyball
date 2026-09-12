#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 4 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int lane = 0; lane < lanes; ++lane)
      if (data[offset + row * stride + lane] != lane) {
        printf("int8mul ball mismatch row=%d lane=%d\n", row, lane);
        exit(1);
      }
  printf("int8mul ball PASS\n");
}
