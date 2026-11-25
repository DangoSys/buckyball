#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *output, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  if (rows != 4 || columns != 16)
    abort();
  for (int row = 0; row < 4; ++row)
    for (int lane = 0; lane < 16; ++lane) {
      uint8_t input = (uint8_t)((row * 16 + lane) * 29 - 128);
      int8_t expected = (int8_t)(input + lane * 17 - 128);
      if (output[offset + row * rs + lane * cs] != expected) {
        printf("FAILED: channel LUT row=%d lane=%d\n", row, lane);
        abort();
      }
    }
  printf("PASSED: sixteen independent INT8 lookup tables\n");
}
