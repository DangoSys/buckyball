#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  const int8_t input[16] = {-128, -127, -96, -64, -32, -1, 0,   1,
                            2,    3,    31,  32,  63,  64, 126, 127};
  if (rows != 2 || columns != 24)
    abort();
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 24; ++column) {
      int8_t expected = (int8_t)((uint8_t)input[column % 16] * 5 + column * 7);
      if (data[offset + row * rs + column * cs] != expected)
        abort();
    }
  printf("PASSED frontend lookup: per-channel tables, signed inputs and "
         "channel tail\n");
}
