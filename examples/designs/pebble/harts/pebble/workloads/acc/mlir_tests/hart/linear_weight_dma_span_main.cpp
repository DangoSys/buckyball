#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t cols, int64_t rs,
                             int64_t cs) {
  if (rows != 8 || cols != 400)
    exit(1);
  for (int row = 0; row < 8; ++row)
    for (int col = 0; col < 400; ++col)
      if (data[offset + row * rs + col * cs] != 163)
        exit(1);
  printf("PASS: strided weight DMA with a page offset and bounded address "
         "spans\n");
}
