// The OpenSBI/Linux timer path: an M-mode MTIP handler raises STIP, the
// delegated supervisor timer interrupt wakes an S-mode wfi taken with SIE
// clear, and an ecall back to M clears STIP.
#include "soc.h"

#define MIP_STIP (1UL << 5)
#define MSTATUS_MPP (3UL << 11)
#define MSTATUS_MPP_S (1UL << 11)
#define SSTATUS_SIE (1UL << 1)
#define SCU_EXIT (*(volatile uint32_t *)0x60000000UL)

static volatile int machine_timer, supervisor_timer, exiting;
static volatile uint32_t result;

static void __attribute__((interrupt("machine"), aligned(4))) machine(void) {
  uint64_t cause = read_csr(mcause);
  if (cause == (MCAUSE_INTERRUPT | 7)) {
    // Hand the tick to S-mode and mask MTIP until the next compare, as OpenSBI
    // does.
    CLINT_MTIMECMP(read_csr(mhartid)) = ~0ULL;
    clear_csr(mie, MIP_MTIP);
    set_csr(mip, MIP_STIP);
    machine_timer++;
    return;
  }
  if (cause == 9 && exiting)
    for (SCU_EXIT = result;;)
      ;
  if (cause == 9) {
    clear_csr(mip, MIP_STIP);
    write_csr(mepc, read_csr(mepc) + 4);
    return;
  }
  for (SCU_EXIT = 0x100 | cause;;)
    ;
}

static void __attribute__((interrupt("supervisor"), aligned(4)))
supervisor(void) {
  if (read_csr(scause) == (MCAUSE_INTERRUPT | 5)) {
    supervisor_timer++;
    asm volatile("ecall");
  }
}

static void __attribute__((noreturn)) supervisor_main(void) {
  // Idle like Linux: wfi with SIE clear must still wake on a pending enabled
  // STIP.
  for (int i = 0; i < 100000 && !supervisor_timer; i++) {
    clear_csr(sstatus, SSTATUS_SIE);
    asm volatile("wfi");
    set_csr(sstatus, SSTATUS_SIE);
  }
  clear_csr(sstatus, SSTATUS_SIE);
  result = machine_timer != 1           ? 1
           : supervisor_timer != 1      ? 2
           : (read_csr(sip) & MIP_STIP) ? 3
                                        : 0;
  exiting = 1;
  asm volatile("ecall");
  __builtin_unreachable();
}

int main(void) {
  // Open all memory to S-mode, as OpenSBI's root domain does.
  write_csr(pmpaddr0, ~0UL);
  write_csr(pmpcfg0, 0x1f);
  write_csr(mtvec, (uintptr_t)machine);
  write_csr(stvec, (uintptr_t)supervisor);
  write_csr(mideleg, MIP_STIP);
  write_csr(sie, MIP_STIP);
  CLINT_MTIMECMP(read_csr(mhartid)) = CLINT_MTIME + 300;
  set_csr(mie, MIP_MTIP | MIP_STIP);
  clear_csr(mstatus, MSTATUS_MPP);
  set_csr(mstatus, MSTATUS_MPP_S);
  write_csr(mepc, (uintptr_t)supervisor_main);
  asm volatile("mret");
  __builtin_unreachable();
}
