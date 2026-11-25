#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  if (rows != 1 || columns != 16)
    abort();
  for (int i = 0; i < 16; ++i) {
    int32_t bias = i == 0 ? INT32_MAX : i == 1 ? INT32_MIN : i - 17;
    uint32_t expected = (uint32_t)bias + 16;
    if ((uint32_t)data[offset + i * cs] != expected)
      abort();
  }
  printf("PASSED: matmul bias fusion with modular I32 addition\n");
}
