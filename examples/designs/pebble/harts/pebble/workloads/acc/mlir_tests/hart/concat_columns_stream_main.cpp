#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  if (rows != 5 || columns != 32)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < columns; ++col)
      if (data[offset + row * rs + col * cs] != (col < 16 ? 1 : 2))
        exit(1);
  printf("PASS: concatenation across full and partial output tiles\n");
}
