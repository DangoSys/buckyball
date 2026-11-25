#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_i32(int32_t *, int32_t *data, int64_t offset,
                          int64_t rows, int64_t cols, int64_t rs, int64_t cs) {
  const int indices[6] = {5, 0, 2, 3, 2, 1};
  if (rows != 6 || cols != 4)
    exit(1);
  for (int row = 0; row < 6; ++row)
    for (int col = 0; col < 4; ++col) {
      int expected = indices[row] * 4 + col;
      if (expected == 23)
        expected = -1;
      if (data[offset + row * rs + col * cs] != expected)
        exit(1);
    }
}

extern "C" void check_f32(float *, float *data, int64_t offset, int64_t rows,
                          int64_t cols, int64_t rs, int64_t cs) {
  if (rows != 2 || cols != 4)
    exit(1);
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 4; ++col)
      if (data[offset + row * rs + col * cs] != (row == 0 ? 8 + col : col))
        exit(1);
  printf("PASS: gathered rows across source boundaries and repeated indices\n");
}
