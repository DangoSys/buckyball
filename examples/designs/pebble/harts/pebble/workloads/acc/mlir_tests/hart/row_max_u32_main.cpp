#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(uint32_t *, uint32_t *data, int64_t offset,
                             int64_t size, int64_t stride) {
  const uint32_t expected[3] = {UINT32_MAX, 0x80000000u, 7};
  if (size != 3)
    exit(1);
  for (int i = 0; i < 3; ++i)
    if (data[offset + i * stride] != expected[i])
      exit(1);
  printf("PASS: unsigned row maxima with ties and high-bit keys\n");
}
