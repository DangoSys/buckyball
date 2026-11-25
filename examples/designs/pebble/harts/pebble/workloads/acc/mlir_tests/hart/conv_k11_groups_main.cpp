#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(float *, float *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t ns,
                             int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 1 || w != 1 || c != 16)
    abort();
  for (int channel = 0; channel < 16; ++channel)
    if (data[offset + channel * cs] != 363.0f) {
      printf("FAILED K11 convolution: channel=%d\n", channel);
      abort();
    }
  printf("PASSED K11 convolution: eleven groups and deferred scales\n");
}
