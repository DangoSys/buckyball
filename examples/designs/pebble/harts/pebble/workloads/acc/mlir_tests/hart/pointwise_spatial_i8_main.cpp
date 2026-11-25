#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t ns, int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 3 || w != 3 || c != 16)
    exit(1);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int channel = 0; channel < 16; ++channel)
        if (data[offset + y * hs + x * ws + channel * cs] !=
            16 * (y * 3 + x + 1) + channel - 17)
          exit(1);
  printf("PASS: spatial pointwise convolution uses direct matrix DMA\n");
}
