#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t ns,
                             int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 1 || w != 128 || c != 64)
    abort();
  for (int x = 0; x < 128; ++x)
    for (int channel = 0; channel < 64; ++channel)
      if (data[offset + x * ws + channel * cs] != (channel < 32 ? 2 : 6))
        abort();
  puts("PASSED: periodic I32 quantization within prepared DMA capacity");
}
