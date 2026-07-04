#include <stdint.h>
#include <stdio.h>

#define SCU_UART_TX ((volatile uint8_t *)0x60020000UL)
#define SCU_UART_RX ((volatile uint8_t *)0x60020004UL)
#define SCU_UART_STATUS ((volatile uint8_t *)0x60020005UL)
#define SCU_UART_RX_VALID 0x01

static void uart_putc(uint8_t ch) { *SCU_UART_TX = ch; }

static uint8_t uart_getc(void) {
  while ((*SCU_UART_STATUS & SCU_UART_RX_VALID) == 0) {
  }
  return *SCU_UART_RX;
}

int main(void) {
  printf("SCU RX echo ready\n");

  for (;;) {
    uint8_t ch = uart_getc();
    uart_putc(ch);
    if (ch == '\n') {
      break;
    }
  }

  printf("SCU RX echo done\n");
  return 0;
}
