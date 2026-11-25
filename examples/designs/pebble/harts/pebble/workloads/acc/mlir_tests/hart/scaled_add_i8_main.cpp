#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  const int8_t expected[16] = {0,   2,    2,   4,    0, -2, -2, -4,
                               127, -128, 120, -120, 0, 3,  -3, 1};
  if (rows != 2 || columns != 16)
    exit(1);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 16; ++column)
      if (data[offset + row * rs + column * cs] != expected[column])
        exit(1);
  printf("PASS: scaled INT8 addition with ties and saturation\n");
}
