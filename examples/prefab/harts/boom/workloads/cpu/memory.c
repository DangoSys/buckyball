// A 256 KiB array, larger than the data cache, written and read back with
// several strides: line refills, write-backs and evictions.
#include <stdint.h>
#include <stdio.h>

#define WORDS (256 * 1024 / 8)
static uint64_t data[WORDS];

static uint64_t pattern(uint64_t i, uint64_t seed) {
  return (i * UINT64_C(0x9e3779b97f4a7c15)) ^ seed;
}

int main(void) {
  for (uint64_t stride = 1; stride <= 64; stride *= 8) {
    for (uint64_t i = 0; i < WORDS; i += stride)
      data[i] = pattern(i, stride);
    for (uint64_t i = 0; i < WORDS; i += stride)
      if (data[i] != pattern(i, stride)) {
        printf("memory FAIL stride=%lu index=%lu\n", (unsigned long)stride,
               (unsigned long)i);
        return 1;
      }
  }
  volatile uint8_t *bytes = (volatile uint8_t *)data;
  for (int i = 0; i < 4096; i++)
    bytes[i * 7] = (uint8_t)i;
  for (int i = 0; i < 4096; i++)
    if (bytes[i * 7] != (uint8_t)i) {
      printf("memory byte FAIL index=%d\n", i);
      return 1;
    }
  printf("memory PASS\n");
  return 0;
}
