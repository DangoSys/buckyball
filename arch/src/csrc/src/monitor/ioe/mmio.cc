#include "ioe/mmio.h"
#include "bdb.h"

#include <cstdio>
#include <cstdint>
#include <unistd.h>

#define SIM_EXIT_ADDR 0x60000000ULL
#define UART_TX_ADDR  0x60020000ULL

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

// Called once per posedge after eval().
// io_mmio_fire is a 1-cycle pulse (harness clock domain = C++ sampling clock).
void mmio_tick() {
  if (!top->io_mmio_fire) return;

  uint64_t addr = top->io_mmio_fire_addr;
  uint64_t data = top->io_mmio_fire_data;

  if (addr == SIM_EXIT_ADDR) {
    int code = (int)(data & 0xFFFFFFFF);
    if (code == 0) fprintf(stderr, "[MMIO] simulation success\n");
    else           fprintf(stderr, "[MMIO] simulation exit code %d\n", code);
    if (uart_fp) fclose(uart_fp);
    sim_exit();
  } else if (addr == UART_TX_ADDR) {
    uart_putchar((char)(data & 0xFF));
  }
}

void mmio_tick_post() {
  // No-op: handshake is now managed entirely in RTL.
}
