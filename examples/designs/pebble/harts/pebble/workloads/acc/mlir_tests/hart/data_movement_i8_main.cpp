#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t cols, int64_t rs,
                             int64_t cs) {
  const int indices[4] = {2, 0, 2, 1};
  const int right[3] = {-128, -64, 127};
  if (rows != 4 || cols != 32)
    exit(1);
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 32; ++col)
      if (data[offset + row * rs + col * cs] !=
          (col < 16 ? indices[row] + 1 : right[indices[row]]))
        exit(1);
  printf("PASS: INT8 channel concatenation and row gathering preserve bytes\n");
}
