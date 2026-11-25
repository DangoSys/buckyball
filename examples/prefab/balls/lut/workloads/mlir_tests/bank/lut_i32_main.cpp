#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  const int32_t expected[8] = {1436, 1436, 1439, 2969, -100, -97, 1433, 1433};
  if (rows != 2 || lanes != 4 || stride != 4 || lane_stride != 1)
    exit(1);
  for (int i = 0; i < 8; ++i)
    if (data[offset + i] != expected[i])
      exit(1);
  puts("lut_i32 PASS");
}
