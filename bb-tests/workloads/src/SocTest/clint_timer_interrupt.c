// A CLINT timer compare raises a machine timer interrupt; rearming clears it.
#include "soc.h"

static volatile uint64_t cause;
static volatile int fired;

static void __attribute__((interrupt("machine"), aligned(4))) handler(void) {
  cause = read_csr(mcause);
  CLINT_MTIMECMP(read_csr(mhartid)) = ~0ULL;
  fired++;
}

int main(void) {
  uint64_t hart = read_csr(mhartid);
  uint64_t start = CLINT_MTIME;
  write_csr(mtvec, (uintptr_t)handler);
  CLINT_MTIMECMP(hart) = start + 200;
  set_csr(mie, MIP_MTIP);
  set_csr(mstatus, MSTATUS_MIE);
  for (int i = 0; i < 100000 && !fired; i++)
    asm volatile("nop");
  clear_csr(mstatus, MSTATUS_MIE);
  clear_csr(mie, MIP_MTIP);
  if (fired != 1)
    return 1;
  if (cause != (MCAUSE_INTERRUPT | 7))
    return 2;
  if (CLINT_MTIME - start < 200)
    return 3;
  if (read_csr(mip) & MIP_MTIP)
    return 4;
  return 0;
}
