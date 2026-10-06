#include "soc.h"
#include <stdio.h>
#define MIP_STIP (1UL << 5)
#define SIE (1UL << 1)
#define SCU_EXIT (*(volatile uint32_t *)0x60000000UL)
static volatile unsigned machine_timer, supervisor_timer, exiting, result;
static volatile unsigned period, overrun, no_foreground, clear_errors;
static volatile uint64_t deadline, foreground, last_foreground;
static volatile uint64_t body_sum, body_max, rearm_sum, rearm_max;
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
    uint64_t start = read_csr(cycle);
    supervisor_timer++;
    if (period) {
      if (foreground == last_foreground)
        no_foreground++;
      last_foreground = foreground;
      deadline = supervisor_timer < 16 ? deadline + period : 0;
    }
    uint64_t arm = read_csr(cycle);
    asm volatile("ecall" ::: "memory");
    uint64_t end = read_csr(cycle), body = end - start, rearm = end - arm;
    body_sum += body;
    rearm_sum += rearm;
    if (body > body_max)
      body_max = body;
    if (rearm > rearm_max)
      rearm_max = rearm;
    if (deadline && read_csr(time) >= deadline)
      overrun++;
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
      machine_timer = supervisor_timer = overrun = no_foreground =
          clear_errors = 0;
      foreground = last_foreground = body_sum = body_max = rearm_sum =
          rearm_max = 0;
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
      printf("timer time_ticks=%u M=%u S=%u clear_errors=%u foreground=%lu "
             "no_foreground=%u\n",
             period, machine_timer, supervisor_timer, clear_errors,
             (unsigned long)foreground, no_foreground);
      printf("timer body_cycles_sum=%lu max=%lu rearm_cycles_sum=%lu max=%lu "
             "overrun=%u\n",
             (unsigned long)body_sum, (unsigned long)body_max,
             (unsigned long)rearm_sum, (unsigned long)rearm_max, overrun);
      result |= clear_errors ? 4 : 0;
      result |= machine_timer != 16 || supervisor_timer != 16 ? 8 : 0;
      result |= !foreground ? 16 : 0;
    }
  for (unsigned delta = 100; delta <= 10000; delta *= 10) {
    deadline = read_csr(time) + delta;
    uint64_t target = deadline;
    asm volatile("ecall" ::: "memory");
    uint64_t returned = read_csr(time);
    deadline = 0;
    asm volatile("ecall" ::: "memory");
    printf("timer short_delta=%u return_minus_deadline=%ld\n", delta,
           (long)(returned - target));
  }
  exiting = 1;
  asm volatile("ecall" ::: "memory");
  __builtin_unreachable();
}
int main(void) {
  write_csr(pmpaddr0, ~0UL);
  write_csr(pmpcfg0, 0x1f);
  write_csr(mcounteren, 3);
  write_csr(scounteren, 3);
  clear_csr(mcountinhibit, 1);
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
