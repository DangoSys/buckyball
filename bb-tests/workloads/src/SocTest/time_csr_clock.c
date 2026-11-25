// TIME follows writable CLINT mtime, independently of writable/inhibited
// mcycle. TIME is a read-only shadow of mtime with no ordering against MMIO
// accesses to the CLINT, so each check waits a bounded number of reads for
// TIME to reflect what mtime holds.
#include "soc.h"

#define READS 1000

int main(void) {
  const uint64_t base = 0x1234567800000000ULL;
  CLINT_MTIME = base;
  uint64_t previous = read_csr(time);
  for (int i = 0; i < READS && previous < base; i++)
    previous = read_csr(time);
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
  // Wait for mtime to tick rather than a count of instructions.
  for (uint64_t tick = CLINT_MTIME; CLINT_MTIME == tick;)
    ;
  after = read_csr(time);
  for (int i = 0; i < READS && after <= before; i++)
    after = read_csr(time);
  if (read_csr(mcycle) != cycles || after <= before)
    return 4;
  clear_csr(mcountinhibit, 1);
  // 32-bit halves also update the same clock observed by TIME.
  *(volatile uint32_t *)(CLINT_BASE + 0xbffc) = 0x76543210;
  uint64_t now = read_csr(time);
  for (int i = 0; i < READS && (now >> 32) != 0x76543210; i++)
    now = read_csr(time);
  if ((now >> 32) != 0x76543210)
    return 5;
  uint64_t mmio = CLINT_MTIME;
  now = read_csr(time);
  for (int i = 0; i < READS && now < mmio; i++)
    now = read_csr(time);
  if (now < mmio || now - mmio > 100000)
    return 6;
  return 0;
}
