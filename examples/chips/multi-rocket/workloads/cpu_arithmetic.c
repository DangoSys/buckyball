#include <stdint.h>
#include <stdio.h>

int main(void) {
  volatile int64_t lhs = -123456789;
  volatile int64_t rhs = 37;
  volatile uint64_t bits = UINT64_C(0xfedcba9876543210);
  if (lhs * rhs != -4567901193LL || lhs / rhs != -3336669 || lhs % rhs != -36 ||
      (bits >> 36) != UINT64_C(0xfedcba9))
    return 1;
  printf("CPU arithmetic passed\n");
  return 0;
}
