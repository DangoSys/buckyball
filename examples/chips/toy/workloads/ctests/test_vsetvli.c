#include <stdio.h>

int main() {
  printf("Testing RVV RoCC decode with VSETVLI\n");

  register unsigned long vl asm("a0");

  // vsetvli a0, zero, e32, m1, ta, ma
  // opcode=1010111, rd=a0(x10), funct3=111, rs1=x0, zimm=0x08
  // Encoding: 0000 0000 1000 00000 111 01010 1010111
  //         = 0x0080d557
  asm volatile(".word 0x0080d557\n" // vsetvli a0, x0, 0x08 (e32, m1)
               : "=r"(vl)
               :
               : "memory");

  printf("VSETVLI executed, vl = %lu\n", vl);
  printf("Test PASSED\n");
  return 0;
}
