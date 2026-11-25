#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  const int8_t expected[32] = {32, 16, 8, 4, 32, 16, 8, 4};
  if (rows != 2 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int i = 0; i < 32; ++i)
    if (data[offset + i] != expected[i])
      exit(1);
  puts("lut_softmax_i8 PASS");
}
