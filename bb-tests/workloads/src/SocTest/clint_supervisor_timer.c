#include "soc.h"
#define MIP_STIP (1UL << 5)
#define SIE (1UL << 1)
#define SCU_EXIT (*(volatile uint32_t *)0x60000000UL)
static volatile unsigned machine_timer, supervisor_timer, exiting, result;
static volatile unsigned period, clear_errors;
static volatile uint64_t deadline, foreground;
static void __attribute__((interrupt("machine"), aligned(4))) machine(void) {
  uint64_t cause = read_csr(mcause);
  if (cause == (MCAUSE_INTERRUPT | 7)) {
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
    if (read_csr(mip) & MIP_STIP)
      clear_errors++;
    CLINT_MTIMECMP(read_csr(mhartid)) = deadline ? deadline : ~0ULL;
    if (deadline)
      set_csr(mie, MIP_MTIP);
    else
      clear_csr(mie, MIP_MTIP);
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
    if (period)
      deadline = supervisor_timer < 16 ? deadline + period : 0;
    asm volatile("ecall" ::: "memory");
  }
}
static void __attribute__((noreturn)) supervisor_main(void) {
  uint64_t limit = read_csr(time) + 1000000;
  while (!(read_csr(sip) & MIP_STIP) && read_csr(time) < limit)
    ;
  if (read_csr(sip) & MIP_STIP) {
    asm volatile("wfi");
    set_csr(sstatus, SIE);
    while (!supervisor_timer && read_csr(time) < limit)
      ;
  }
  clear_csr(sstatus, SIE);
  result = machine_timer != 1                           ? 1
           : supervisor_timer != 1                      ? 2
           : (read_csr(sip) & MIP_STIP) || clear_errors ? 3
                                                        : 0;
  if (!result)
    for (period = 20; period <= 50; period += 30) {
      machine_timer = supervisor_timer = clear_errors = 0;
      foreground = 0;
      deadline = read_csr(time) + period;
      limit = deadline + 32UL * period;
      asm volatile("ecall" ::: "memory");
      set_csr(sstatus, SIE);
      for (unsigned i = 0;
           i < 2000000 && supervisor_timer < 16 && read_csr(time) < limit; i++)
        foreground++;
      clear_csr(sstatus, SIE);
      deadline = 0;
      asm volatile("ecall" ::: "memory");
      result |= clear_errors ? 4 : 0;
      result |= machine_timer != 16 || supervisor_timer != 16 ? 8 : 0;
      result |= !foreground ? 16 : 0;
    }
  exiting = 1;
  asm volatile("ecall" ::: "memory");
  __builtin_unreachable();
}
int main(void) {
  write_csr(pmpaddr0, ~0UL);
  write_csr(pmpcfg0, 0x1f);
  write_csr(mcounteren, 3);
  write_csr(mtvec, (uintptr_t)machine);
  write_csr(stvec, (uintptr_t)supervisor);
  write_csr(mideleg, MIP_STIP);
  write_csr(sie, MIP_STIP);
  CLINT_MTIMECMP(read_csr(mhartid)) = CLINT_MTIME + 300;
  set_csr(mie, MIP_MTIP | MIP_STIP);
  clear_csr(mstatus, (3UL << 11) | SIE);
  set_csr(mstatus, 1UL << 11);
  write_csr(mepc, (uintptr_t)supervisor_main);
  asm volatile("mret");
  __builtin_unreachable();
}
