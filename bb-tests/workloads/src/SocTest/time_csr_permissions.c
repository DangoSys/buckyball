// TIME is readable in M/S/U according to counter enables, but writes always
// trap.
#include "soc.h"
#define EXIT (*(volatile uint32_t *)0x60000000UL)
static volatile unsigned illegal, action;
static void __attribute__((noreturn)) user(void);
static void __attribute__((noreturn)) finish(unsigned result) {
  for (EXIT = result;;)
    ;
}
static void __attribute__((interrupt("machine"), aligned(4))) trap(void) {
  uint64_t cause = read_csr(mcause);
  if (cause == 2) {
    illegal++;
    write_csr(mepc, read_csr(mepc) + 4);
    return;
  }
  if (cause != 8 && cause != 9)
    finish(0x100 | cause);
  if (action == 3) {
    write_csr(mcounteren, 2);
    write_csr(scounteren, 0);
    clear_csr(mstatus, 3UL << 11);
    write_csr(mepc, (uintptr_t)user);
    return;
  }
  if (action >= 4)
    finish(action - 4);
  write_csr(mcounteren, action == 2 ? 0 : 2);
  write_csr(scounteren, 2);
  write_csr(mepc, read_csr(mepc) + 4);
}
static void request(unsigned value) {
  action = value;
  asm volatile("ecall" ::: "memory");
}
static void readonly(void) {
  unsigned before = illegal;
  asm volatile("csrw time, zero; csrsi time, 1; csrci time, 1");
  if (illegal != before + 3)
    request(5);
}
static uint64_t clint_time(void) {
  uint64_t before = CLINT_MTIME, value;
  asm volatile("fence iorw, iorw; csrr %0, time; fence iorw, iorw"
               : "=r"(value)::"memory");
  uint64_t after = CLINT_MTIME;
  return value >> 32 == 0x12345678 && value + 1 >= before && value <= after
             ? value
             : 0;
}
static void __attribute__((noreturn)) user(void) {
  unsigned before = illegal;
  (void)read_csr(time); // mcounteren allows, scounteren denies.
  if (illegal != before + 1)
    request(6);
  request(1);
  uint64_t value = clint_time();
  if (!value || illegal != before + 1)
    request(7);
  readonly();
  before = illegal;
  request(2);
  (void)read_csr(time); // scounteren allows, mcounteren denies.
  if (illegal != before + 1)
    request(8);
  request(4);
  __builtin_unreachable();
}
static void __attribute__((noreturn)) supervisor(void) {
  unsigned before = illegal;
  (void)read_csr(time);
  if (illegal != before + 1)
    request(9);
  request(1);
  uint64_t value = clint_time();
  if (!value || illegal != before + 1)
    request(10);
  readonly();
  request(3);
  __builtin_unreachable();
}
int main(void) {
  write_csr(mtvec, (uintptr_t)trap);
  write_csr(pmpaddr0, ~0UL);
  write_csr(pmpcfg0, 0x1f);
  write_csr(mcounteren, 0);
  write_csr(scounteren, 0);
  CLINT_MTIME = 0x1234567800000000ULL;
  if (!read_csr(time))
    return 7;
  readonly();
  clear_csr(mstatus, 3UL << 11);
  set_csr(mstatus, 1UL << 11);
  write_csr(mepc, (uintptr_t)supervisor);
  asm volatile("mret");
  __builtin_unreachable();
}
