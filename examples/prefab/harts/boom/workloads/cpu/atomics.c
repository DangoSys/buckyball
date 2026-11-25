// AMOs and LR/SC on cacheable memory.
#include <stdint.h>
#include <stdio.h>

static volatile uint64_t word = 5;
static volatile uint32_t half = 7;

int main(void) {
  uint64_t old;
  asm volatile("amoadd.d %0, %2, (%1)"
               : "=r"(old)
               : "r"(&word), "r"(10)
               : "memory");
  if (old != 5 || word != 15) {
    printf("atomics amoadd FAIL\n");
    return 1;
  }
  asm volatile("amoswap.d %0, %2, (%1)"
               : "=r"(old)
               : "r"(&word), "r"(3)
               : "memory");
  if (old != 15 || word != 3) {
    printf("atomics amoswap FAIL\n");
    return 1;
  }
  uint32_t old32;
  asm volatile("amomax.w %0, %2, (%1)"
               : "=r"(old32)
               : "r"(&half), "r"(9)
               : "memory");
  if (old32 != 7 || half != 9) {
    printf("atomics amomax FAIL\n");
    return 1;
  }
  uint64_t value, fail = 1;
  for (int i = 0; i < 16 && fail; i++)
    asm volatile("lr.d %0, (%2)\n\taddi %0, %0, 1\n\tsc.d %1, %0, (%2)"
                 : "=&r"(value), "=&r"(fail)
                 : "r"(&word)
                 : "memory");
  if (fail || word != 4) {
    printf("atomics lr/sc FAIL\n");
    return 1;
  }
  printf("atomics PASS\n");
  return 0;
}
