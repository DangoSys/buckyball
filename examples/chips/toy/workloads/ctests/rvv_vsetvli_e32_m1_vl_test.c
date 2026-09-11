#include <params.h>
#include <stdio.h>

int main(void) {
  register unsigned long vl asm("a0");
  asm volatile(".option push\n"
               ".option arch, +v\n"
               "vsetvli %0, zero, e32, m1, ta, ma\n"
               ".option pop\n"
               : "=r"(vl)
               :
               : "memory");
  unsigned long expected = RVV_VLEN_BITS / 32;
  if (vl != expected) {
    printf("vsetvli e32,m1 FAIL expected=%lu actual=%lu\n", expected, vl);
    return 1;
  }
  printf("vsetvli e32,m1 PASS vl=%lu\n", vl);
  return 0;
}
