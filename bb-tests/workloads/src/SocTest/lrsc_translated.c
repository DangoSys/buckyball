// An S-mode LR/SC under Sv39 completes even when the reserved line and the PTE
// its SC reads map to the same direct-mapped L1 index: here the root PTE and
// the target both sit in line 0 of their pages, so a translation walk between
// LR and SC must not evict the reservation.
#include "soc.h"

#define MSTATUS_MPP (3UL << 11)
#define MSTATUS_MPP_S (1UL << 11)
#define SCU_EXIT (*(volatile uint32_t *)0x60000000UL)
#define PTE_VRWXAD 0xcfUL

static uint64_t root[512] __attribute__((aligned(4096)));
static volatile uint32_t target[1024] __attribute__((aligned(4096)));
static volatile uint32_t result;

static void __attribute__((interrupt("machine"), aligned(4))) machine(void) {
  // The only expected trap is the S-mode ecall that reports the result.
  uint64_t cause = read_csr(mcause);
  for (SCU_EXIT = cause == 9 ? result : 0x100 | cause;;)
    ;
}

static void __attribute__((noreturn)) supervisor_main(void) {
  volatile uint32_t *word =
      &target[5]; // upper word of a doubleword, as in atomic_add_unless
  uint32_t old;
  uint64_t fail = 1;
  for (int tries = 0; tries < 16 && fail; tries++)
    asm volatile("lr.w %0, (%2)\n\taddi %0, %0, 1\n\tsc.w.rl %1, %0, (%2)"
                 : "=&r"(old), "=&r"(fail)
                 : "r"(word)
                 : "memory");
  result = fail ? 1 : *word != 1 ? 2 : 0;
  asm volatile("ecall");
  __builtin_unreachable();
}

int main(void) {
  // One 1-GiB identity leaf covers DRAM; the SCU stays reachable from M-mode.
  root[2] = (0x80000000UL >> 12) << 10 | PTE_VRWXAD;
  write_csr(pmpaddr0, ~0UL);
  write_csr(pmpcfg0, 0x1f);
  write_csr(mtvec, (uintptr_t)machine);
  write_csr(satp, (8UL << 60) | ((uintptr_t)root >> 12));
  asm volatile("sfence.vma" ::: "memory");
  clear_csr(mstatus, MSTATUS_MPP);
  set_csr(mstatus, MSTATUS_MPP_S);
  write_csr(mepc, (uintptr_t)supervisor_main);
  asm volatile("mret");
  __builtin_unreachable();
}
