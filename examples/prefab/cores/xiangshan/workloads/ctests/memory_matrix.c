#include <stdint.h>

static volatile int32_t a[32][32], b[32][16], c[32][16];
static volatile uint64_t atomic_word;
static volatile uint64_t upgrade_word[8] __attribute__((aligned(64))) = {7};

int main(void) {
  for (int i = 0; i < 32; ++i)
    for (int k = 0; k < 32; ++k)
      a[i][k] = (i * 3 + k * 5) % 13 - 6;
  for (int k = 0; k < 32; ++k)
    for (int j = 0; j < 16; ++j)
      b[k][j] = (k * 7 + j * 2) % 11 - 5;
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 16; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < 32; ++k)
        sum += a[i][k] * b[k][j];
      c[i][j] = sum;
    }
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 16; ++j) {
      int32_t expected = 0;
      for (int k = 0; k < 32; ++k)
        expected += ((i * 3 + k * 5) % 13 - 6) * ((k * 7 + j * 2) % 11 - 5);
      if (c[i][j] != expected)
        return 2;
    }
  uint64_t old, value = 19;
  __asm__ volatile("amoadd.d.aqrl %0, %2, (%1)"
                   : "=r"(old)
                   : "r"(&atomic_word), "r"(value)
                   : "memory");
  if (old != 0 || atomic_word != 19)
    return 3;
  if (upgrade_word[0] != 7)
    return 5;
  upgrade_word[0] = 23;
  if (upgrade_word[0] != 23)
    return 6;
  uint64_t before, after;
  __asm__ volatile("rdtime %0" : "=r"(before));
  __asm__ volatile("rdtime %0" : "=r"(after));
  if (before == 0 || after <= before)
    return 4;
  return 0;
}
