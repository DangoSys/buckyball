#include <stdint.h>
#include <stdio.h>

int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  printf("multi-rocket hart %u: hello\n", (unsigned)hartid);
  return 0;
}
