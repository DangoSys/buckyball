// Setting a hart's CLINT msip raises a machine software interrupt; clearing it
// drops the pending bit.
#include "soc.h"

static volatile uint64_t cause;
static volatile int fired;

static void __attribute__((interrupt("machine"), aligned(4))) handler(void) {
  cause = read_csr(mcause);
  CLINT_MSIP(read_csr(mhartid)) = 0;
  fired++;
}

int main(void) {
  uint64_t hart = read_csr(mhartid);
  write_csr(mtvec, (uintptr_t)handler);
  if (read_csr(mip) & MIP_MSIP)
    return 1;
  set_csr(mie, MIP_MSIP);
  set_csr(mstatus, MSTATUS_MIE);
  CLINT_MSIP(hart) = 1;
  for (int i = 0; i < 10000 && !fired; i++)
    asm volatile("nop");
  clear_csr(mstatus, MSTATUS_MIE);
  clear_csr(mie, MIP_MSIP);
  if (fired != 1)
    return 2;
  if (cause != (MCAUSE_INTERRUPT | 3))
    return 3;
  if (CLINT_MSIP(hart) != 0 || (read_csr(mip) & MIP_MSIP))
    return 4;
  return 0;
}
