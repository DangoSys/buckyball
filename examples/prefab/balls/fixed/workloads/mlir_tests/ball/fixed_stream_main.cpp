#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  static const int32_t expected[16] = {
      0,   2,   0,    -2,   8,          10,          -8, -10,
      127, 128, -127, -128, 1073741824, -1073741824, 8,  -8};
  if (rows != 4 || columns != 4)
    abort();
  for (int i = 0; i < 16; ++i)
    if (data[offset + i / 4 * rs + i % 4 * cs] != expected[i])
      abort();
  printf("PASSED: fixed streaming coefficients\n");
}
