#include "ioe/mmio.h"
#include "bdb.h"

#include <cstdint>
#include <cstdio>
#include <unistd.h>

// SCU DPI-C implementation for Verilator simulation.
// These functions are called from the RTL when software writes to SCU
// registers.

static FILE *uart_fp = nullptr;

static void uart_putchar(char ch) {
  if (!uart_fp) {
    const char *path = stdout_path ? stdout_path : "stdout.log";
    uart_fp = fopen(path, "w");
  }
  if (uart_fp) {
    fputc(ch, uart_fp);
    fflush(uart_fp);
  }
  if (raw_stdout_fd >= 0) {
    write(raw_stdout_fd, &ch, 1);
  }
}

// DPI-C function: called from RTL when software writes to SCU UART TX register
// (address 0x6002_0000, offset +0x20000 from SCU base)
extern "C" void scu_uart_write(uint32_t hart_id, uint32_t ch) {
  (void)hart_id;
  uart_putchar((char)(ch & 0xff));
}

// DPI-C function: called from RTL when software writes to SCU sim_exit register
// (address 0x6000_0000, offset +0x00000 from SCU base)
extern "C" void scu_sim_exit(uint32_t hart_id, uint32_t code) {
  if (code == 0)
    fprintf(stderr, "[SCU] hart %u: simulation success\n", hart_id);
  else
    fprintf(stderr, "[SCU] hart %u: simulation exit code %u\n", hart_id, code);

  if (uart_fp)
    fclose(uart_fp);

  sim_exit();
}
