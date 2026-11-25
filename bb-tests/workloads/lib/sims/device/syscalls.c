#include <stdint.h>

/* The MMIO window of the calling hart (crt0 writes its exit code there). */
static volatile uint32_t *window(uintptr_t offset) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return (volatile uint32_t *)(0x60000000UL + (hart << 18) + offset);
}
#define MMIO_SIM_EXIT window(0)
#define MMIO_UART_TX window(0x20000)

int _write(int fd, const char *buf, int len) {
  (void)fd;
  for (int i = 0; i < len; ++i) {
    *MMIO_UART_TX = (uint32_t)(unsigned char)buf[i];
  }
  return len;
}

void __attribute__((noreturn)) _exit(int code) {
  *MMIO_SIM_EXIT = (uint32_t)code;
  while (1) {
  }
}
