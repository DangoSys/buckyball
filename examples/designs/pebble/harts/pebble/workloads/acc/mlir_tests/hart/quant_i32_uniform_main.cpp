#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  if (rows != 1 || columns != 32)
    abort();
  for (int i = 0; i < 32; ++i) {
    int expected = i / 2 + (i % 4 == 3);
    if (data[offset + i * cs] != expected)
      abort();
  }
  printf("PASSED: uniform quantization across channel blocks\n");
}
