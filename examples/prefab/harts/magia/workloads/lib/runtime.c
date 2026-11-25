#include <stdint.h>

volatile uint32_t bb_started __attribute__((section(".data.bb_started"))) = 0;
static volatile uint32_t results[16];

void bb_exit(int code) {
  uint32_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  results[hart] = UINT32_C(0x80000000) | (uint32_t)code;
  asm volatile("fence rw, rw" : : : "memory");
  if (hart == 0) {
    uint32_t failure = 0;
    for (unsigned i = 0; i < 16; ++i) {
      uint32_t result;
      do {
        result = results[i];
      } while (!(result & UINT32_C(0x80000000)));
      failure |= result != UINT32_C(0x80000000);
    }
    asm volatile("fence rw, rw" : : : "memory");
    *(volatile uint32_t *)(uintptr_t)MAGIA_MAILBOX_BASE = failure;
  }
  for (;;)
    asm volatile("wfi");
}
