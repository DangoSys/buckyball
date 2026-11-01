#include <stdint.h>
#include <stdio.h>

int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  printf("piplus control core: hart %u runs\n", (unsigned)hart);
  return 0;
}
