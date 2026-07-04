#ifndef GOBAN_SCU_H
#define GOBAN_SCU_H

#include <stdint.h>

#define SCU_BASE 0x60000000UL
#define SCU_STRIDE 0x40000UL
#define SCU_UART_TX_OFF 0x20000UL
#define SCU_UART_RX_OFF 0x20004UL
#define SCU_UART_STATUS_OFF 0x20005UL
#define SCU_READY_OFF 0x20006UL
#define SCU_UART_RX_VALID 0x01

static inline volatile uint8_t *scu_reg8(int hart, uintptr_t off) {
  return (volatile uint8_t *)(SCU_BASE + (uintptr_t)hart * SCU_STRIDE + off);
}

static inline void scu_io_fence(void) {
  asm volatile("fence iorw, iorw" ::: "memory");
}

static inline void scu_set_ready(int hart, uint8_t value) {
  scu_io_fence();
  *scu_reg8(hart, SCU_READY_OFF) = value;
  scu_io_fence();
}

static inline uint8_t scu_get_ready(int hart) {
  uint8_t value = *scu_reg8(hart, SCU_READY_OFF);
  scu_io_fence();
  return value;
}

static inline void scu_putc(int hart, char ch) {
  *scu_reg8(hart, SCU_UART_TX_OFF) = (uint8_t)ch;
}

static inline void scu_poll_pause(void) {
  for (int i = 0; i < 256; ++i) {
    asm volatile("nop" ::: "memory");
  }
}

static inline void scu_settle(void) {
  for (int i = 0; i < 256; ++i) {
    scu_poll_pause();
  }
}

static inline uint8_t scu_getc(int hart) {
  while ((*scu_reg8(hart, SCU_UART_STATUS_OFF) & SCU_UART_RX_VALID) == 0) {
    scu_poll_pause();
  }
  return *scu_reg8(hart, SCU_UART_RX_OFF);
}

static inline void scu_puts(int hart, const char *s) {
  while (*s != '\0') {
    scu_putc(hart, *s++);
  }
}

static inline void scu_put_u32(int hart, uint32_t value) {
  char buf[10];
  int n = 0;

  if (value == 0) {
    scu_putc(hart, '0');
    return;
  }

  while (value != 0) {
    buf[n++] = (char)('0' + value % 10);
    value /= 10;
  }
  while (n > 0) {
    scu_putc(hart, buf[--n]);
  }
}

#endif // GOBAN_SCU_H
