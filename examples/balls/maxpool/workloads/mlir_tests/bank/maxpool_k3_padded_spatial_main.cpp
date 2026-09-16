#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  constexpr int side = 8;
  if (rows != BANK_LINES || lanes != BANK_WIDTH / 8 || stride != lanes ||
      lane_stride != 1)
    exit(1);
  for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x) {
      int8_t expected = -128;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int iy = y + ky;
          int ix = x + kx;
          if (iy >= 0 && iy < side && ix >= 0 && ix < side) {
            int8_t value = (iy * side + ix) % 31 - 15;
            if (value > expected)
              expected = value;
          }
        }
      int row = y * side + x;
      for (int lane = 0; lane < lanes; ++lane)
        if (data[offset + row * stride + lane] != expected) {
          printf("maxpool K3 mismatch row=%d lane=%d\n", row, lane);
          exit(1);
        }
    }
  printf("maxpool K3 padded spatial PASS\n");
}
