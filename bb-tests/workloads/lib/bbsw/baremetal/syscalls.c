#include <stdint.h>

#define MMIO_SIM_EXIT ((volatile uint32_t *)0x60000000UL)
#define MMIO_UART_TX ((volatile uint32_t *)0x60020000UL)

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
