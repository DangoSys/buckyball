// TIME follows writable CLINT mtime, independently of writable/inhibited
// mcycle.
#include "soc.h"

int main(void) {
  const uint64_t base = 0x1234567800000000ULL;
  CLINT_MTIME = base;
  uint64_t previous = read_csr(time);
  if (previous < base || previous - base > 100000)
    return 1;
  for (int i = 0; i < 32; i++) {
    uint64_t now = read_csr(time);
    if (now < previous)
      return 2;
    previous = now;
  }
  uint64_t before = read_csr(time);
  write_csr(mcycle, 0);
  uint64_t after = read_csr(time);
  if (after < before || read_csr(mcycle) >= base)
    return 3;
  set_csr(mcountinhibit, 1);
  uint64_t cycles = read_csr(mcycle);
  before = read_csr(time);
  asm volatile(".rept 2048; nop; .endr");
  after = read_csr(time);
  if (read_csr(mcycle) != cycles || after <= before)
    return 4;
  clear_csr(mcountinhibit, 1);
  // 32-bit halves also update the same clock observed by TIME.
  *(volatile uint32_t *)(CLINT_BASE + 0xbffc) = 0x76543210;
  uint64_t now = read_csr(time);
  if ((now >> 32) != 0x76543210)
    return 5;
  uint64_t mmio = CLINT_MTIME;
  now = read_csr(time);
  if (now < mmio || now - mmio > 100000)
    return 6;
  return 0;
}
