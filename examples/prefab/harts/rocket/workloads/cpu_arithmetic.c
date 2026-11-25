// Integer ALU, multiplier and divider of the hart.
#include <stdint.h>
#include <stdio.h>

int main(void) {
  volatile int64_t lhs = -123456789;
  volatile int64_t rhs = 37;
  volatile uint64_t bits = UINT64_C(0xfedcba9876543210);
  if (lhs * rhs != -4567901193LL || lhs / rhs != -3336669 || lhs % rhs != -36 ||
      (bits >> 36) != UINT64_C(0xfedcba9)) {
    printf("cpu_arithmetic FAIL\n");
    return 1;
  }
  volatile uint64_t a = UINT64_C(0xffffffffffffffff);
  volatile uint64_t b = UINT64_C(0x123456789abcdef1);
  unsigned __int128 product = (unsigned __int128)a * b;
  if ((uint64_t)(product >> 64) != UINT64_C(0x123456789abcdef0) ||
      (uint64_t)product != UINT64_C(0xedcba9876543210f)) {
    printf("cpu_arithmetic mulhu FAIL\n");
    return 1;
  }
  printf("cpu_arithmetic PASS\n");
  return 0;
}
