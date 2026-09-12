#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != BANK_LINES / 4 || lanes != BANK_WIDTH / 8 || stride != lanes ||
      lane_stride != 1)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int lane = 0; lane < lanes; ++lane) {
      int expected = row * lanes + lane;
      if (expected > 127)
        expected = 127;
      if (data[offset + row * stride + lane] != expected) {
        printf("quant f32 bank mismatch row=%d lane=%d\n", row, lane);
        exit(1);
      }
    }
  printf("quant f32 bank PASS rows=%d\n", BANK_LINES);
}
