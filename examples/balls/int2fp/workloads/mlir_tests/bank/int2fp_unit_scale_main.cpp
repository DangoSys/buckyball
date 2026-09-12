#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(float *, float *data, int64_t offset, int64_t rows,
                             int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != BANK_LINES || lanes != BANK_WIDTH / 32 || stride != lanes ||
      lane_stride != 1)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int lane = 0; lane < lanes; ++lane)
      if (data[offset + row * stride + lane] != (float)(row - lane)) {
        printf("int2fp bank mismatch row=%d lane=%d\n", row, lane);
        exit(1);
      }
  printf("int2fp bank PASS rows=%d\n", BANK_LINES);
}
