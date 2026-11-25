#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  static const int32_t expected[16] = {INT32_MAX, INT32_MIN, 0,  2,  0,   -2,
                                       2,         4,         -2, -4, 100, -100,
                                       11,        21,        31, 41};
  if (rows != 4 || columns != 4)
    abort();
  for (int i = 0; i < 16; ++i)
    if (data[offset + (i / 4) * rs + (i % 4) * cs] != expected[i])
      abort();
  printf("PASSED: fixed mean MLIR\n");
}
