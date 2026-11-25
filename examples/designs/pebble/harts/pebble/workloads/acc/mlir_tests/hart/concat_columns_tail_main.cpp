#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_f32(float *, float *data, int64_t offset, int64_t rows,
                          int64_t columns, int64_t rs, int64_t cs) {
  const float expected[] = {1,  2,   3, 4,   .5,  1,   -4,  -3,  -2,
                            -1, .25, 2, 2.5, 3.5, 4.5, 5.5, .75, 3};
  if (rows != 3 || columns != 6)
    exit(1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < columns; ++c)
      if (data[offset + r * rs + c * cs] != expected[r * columns + c])
        exit(1);
}

extern "C" void check_i8(int8_t *, int8_t *data, int64_t offset, int64_t rows,
                         int64_t columns, int64_t rs, int64_t cs) {
  const int8_t expected[] = {-128, 0, 127, 4, 5, -3, -2, -1,
                             6,    7, 1,   2, 3, 8,  9};
  if (rows != 3 || columns != 5)
    exit(1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < columns; ++c)
      if (data[offset + r * rs + c * cs] != expected[r * columns + c])
        exit(1);
  printf("PASS: partial-row FP32 and INT8 concatenation\n");
}
