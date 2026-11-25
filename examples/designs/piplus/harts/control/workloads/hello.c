#include <stdint.h>
#include <stdio.h>

int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  printf("piplus control hart: hart %u runs\n", (unsigned)hartid);
  return 0;
}
