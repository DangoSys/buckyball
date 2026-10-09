#include <stddef.h>
#include <stdint.h>

#define REG32(off) (*(volatile uint32_t *)(uintptr_t)(0x60000000UL + (off)))
#define REG64(off) (*(volatile uint64_t *)(uintptr_t)(0x60000000UL + (off)))
#define BB(fn, a, b)                                                           \
  __asm__ volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"                          \
                   :                                                           \
                   : "r"((uint64_t)(a)), "r"((uint64_t)(b)), "i"(fn)           \
                   : "memory")
static void fence_bb(void) {
  BB(0, 0, 0);
  __asm__ volatile("fence rw,rw; fence.i" ::: "memory");
}
static void move_in(const void *p, unsigned bank, unsigned lines) {
  __asm__ volatile("fence.i" ::: "memory");
  BB(33, ((uint64_t)bank << 20) | ((uint64_t)lines << 30),
     (uintptr_t)p | (1UL << 39));
}
static void move_out(void *p, unsigned bank, unsigned lines) {
  __asm__ volatile("fence.i" ::: "memory");
  BB(16, bank | ((uint64_t)lines << 30), (uintptr_t)p | (1UL << 39));
}
static void putchar_uart(char c) {
  while (!(REG32(4) & 1)) {
  }
  *(volatile uint8_t *)(uintptr_t)0x60000000UL = c;
}
static void puts_uart(const char *s) {
  while (*s)
    putchar_uart(*s++);
}
static void hex(uint64_t v) {
  for (int shift = 60; shift >= 0; shift -= 4)
    putchar_uart("0123456789abcdef"[(v >> shift) & 15]);
}
static void fail(const char *what, unsigned i, uint64_t got,
                 uint64_t expected) {
  puts_uart("FAIL ");
  puts_uart(what);
  puts_uart(" index=");
  hex(i);
  puts_uart(" got=");
  hex(got);
  puts_uart(" expected=");
  hex(expected);
  puts_uart("\r\n");
  REG32(16) = 1;
  for (;;)
    __asm__ volatile("wfi");
}
void trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
  puts_uart("TRAP cause=");
  hex(cause);
  puts_uart(" pc=");
  hex(pc);
  puts_uart(" value=");
  hex(value);
  puts_uart("\r\n");
  REG32(16) = 2;
  for (;;)
    __asm__ volatile("wfi");
}
static int8_t a[8 * 16] __attribute__((aligned(64)));
static int8_t b[16 * 16] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static int32_t addend[128] __attribute__((aligned(64)));
static int32_t result[128] __attribute__((aligned(64)));
static int32_t reference[128] __attribute__((aligned(64)));
static volatile uint64_t memory_test[1024] __attribute__((aligned(64)));
static void run_tests(unsigned seed) {
  puts_uart("RUN seed=");
  hex(seed);
  puts_uart("\r\n");
  for (unsigned i = 0; i < 1024; ++i)
    memory_test[i] = 0x9e3779b97f4a7c15UL * (i + seed + 1);
  for (unsigned i = 0; i < 1024; ++i) {
    uint64_t want = 0x9e3779b97f4a7c15UL * (i + seed + 1);
    if (memory_test[i] != want)
      fail("RAM", i, memory_test[i], want);
  }
  REG64(24) = 0x123456789abcdef0UL;
  *(volatile uint8_t *)(uintptr_t)0x6000001bUL = 0x55;
  if (REG64(24) != 0x1234567855bcdef0UL)
    fail("MMIO", 0, REG64(24), 0x1234567855bcdef0UL);
  puts_uart("PASS CPU_RAM_MMIO\r\n");
  for (unsigned i = 0; i < 128; ++i)
    a[i] = (int)((i * 13 + seed * 7) % 31) - 15;
  for (unsigned i = 0; i < 256; ++i)
    b[i] = (int)((i * 17 + seed * 3) % 29) - 14;
  for (unsigned i = 0; i < 16; ++i)
    bias[i] = (int)i * 37 - 250;
  for (unsigned i = 0; i < 128; ++i) {
    int32_t sum = bias[i % 16];
    for (unsigned k = 0; k < 16; ++k)
      sum += (int32_t)a[i / 16 * 16 + k] * b[k * 16 + i % 16];
    reference[i] = sum;
    addend[i] = (int)i * 11 - 700;
    result[i] = 0x55555555;
  }
  for (unsigned bank = 0; bank < 6; ++bank)
    BB(32, bank, (1 << 10) | (1 << 5) | 1);
  fence_bb();
  move_in(bias, 0, 4);
  move_in(a, 1, 8);
  move_in(b, 2, 16);
  BB(17, 4UL << 30, 0);
  BB(65, 1UL | (2UL << 10) | (3UL << 20) | (16UL << 30),
     8UL | (16UL << 12) | (3UL << 24));
  move_out(result, 3, 32);
  fence_bb();
  for (unsigned i = 0; i < 128; ++i)
    if (result[i] != reference[i])
      fail("SMATMUL", i, result[i], reference[i]);
  puts_uart("PASS SMATMUL_8x16x16\r\n");
  move_in(addend, 4, 32);
  BB(72, 3UL | (4UL << 10) | (5UL << 20) | (32UL << 30), 0);
  move_out(result, 5, 32);
  fence_bb();
  for (unsigned i = 0; i < 128; ++i) {
    reference[i] += addend[i];
    if (result[i] != reference[i])
      fail("MATADD", i, result[i], reference[i]);
  }
  puts_uart("PASS MATADD_I32\r\n");
  BB(50, 5UL | (32UL << 30), 32);
  move_out(result, 5, 32);
  fence_bb();
  uint64_t checksum = 0;
  for (unsigned i = 0; i < 128; ++i) {
    if (reference[i] < 0)
      reference[i] = 0;
    if (result[i] != reference[i])
      fail("RELU", i, result[i], reference[i]);
    checksum += (uint32_t)result[i] * (uint64_t)(i + 1);
  }
  for (unsigned bank = 0; bank < 6; ++bank)
    BB(32, bank, 0);
  fence_bb();
  puts_uart("PASS RELU_I32 checksum=");
  hex(checksum);
  puts_uart("\r\n");
  puts_uart("PASS PEBBLE_A7_FULL_CHIP\r\n");
  REG32(16) = 0;
}
int main(void) {
  /* Enable the custom extension state in mstatus before issuing RoCC. */
  __asm__ volatile("csrs mstatus,%0" ::"r"(3UL << 15) : "memory");
  puts_uart("PEBBLE_A7 RV64IMAC 50MHz\r\n");
  unsigned seed = 1;
  run_tests(seed);
  puts_uart("READY: r reruns with new data; other bytes echo\r\n");
  for (;;) {
    uint32_t rx = REG32(8);
    if (rx >> 31) {
      char ch = (char)rx;
      if (ch == 'r')
        run_tests(++seed);
      else
        putchar_uart(ch);
    }
  }
}
