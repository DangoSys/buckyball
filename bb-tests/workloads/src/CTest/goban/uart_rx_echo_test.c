/*
 * Goban multi-hart SCU UART RX echo test.
 *
 * Every hart announces itself, then waits on its own SCU UART RX queue. Bytes
 * received by hart N are echoed through hart N's SCU UART TX.
 */

#include "goban.h"
#include "scu.h"

#define NCORES 4
#ifndef NTILES
#define NTILES 64
#endif
#ifndef HIDDEN_HART_BASE
#define HIDDEN_HART_BASE 64
#endif

static int expected_hart_id(int tile, int cid) {
  if (cid == 0) {
    return tile;
  }
  return HIDDEN_HART_BASE + tile * (NCORES - 1) + (cid - 1);
}

static const char *hart_role(int hart) {
  if (hart < HIDDEN_HART_BASE) {
    return "visible";
  }
  return "hidden";
}

static void print_ready(int hart, int tile, int cid) {
  scu_puts(hart, "[hart ");
  scu_put_u32(hart, (uint32_t)hart);
  scu_puts(hart, "] ");
  scu_puts(hart, hart_role(hart));
  scu_puts(hart, " t=");
  scu_put_u32(hart, (uint32_t)tile);
  scu_puts(hart, " c=");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, " ready\n");
}

static void wait_all_ready(int hart) {
  scu_set_ready(hart, 1);

  if (hart == 0) {
    int ready = 0;
    while (!ready) {
      ready = 1;
      for (int i = 0; i < NTILES * NCORES; ++i) {
        if (scu_get_ready(i) != 1) {
          ready = 0;
        }
      }
      scu_poll_pause();
    }

    for (int i = 0; i < NTILES * NCORES; ++i) {
      scu_set_ready(i, 2);
    }
    return;
  }

  while (scu_get_ready(hart) != 2) {
    scu_poll_pause();
  }
}

int main(void) {
  int hart = bb_get_hart_id();
  int tile = bb_get_tile_id();
  int cid = bb_get_tile_core_id();

  if (hart < 0 || hart >= NTILES * NCORES || tile < 0 || tile >= NTILES ||
      cid < 0 || cid >= NCORES) {
    scu_puts(hart, "[hart ");
    scu_put_u32(hart, (uint32_t)hart);
    scu_puts(hart, "] invalid ids\n");
    while (1) {
      asm volatile("wfi");
    }
  }

  if (hart != expected_hart_id(tile, cid)) {
    scu_puts(hart, "[hart ");
    scu_put_u32(hart, (uint32_t)hart);
    scu_puts(hart, "] unexpected mapping\n");
    while (1) {
      asm volatile("wfi");
    }
  }

  wait_all_ready(hart);
  print_ready(hart, tile, cid);

  while (1) {
    uint8_t ch = scu_getc(hart);
    scu_putc(hart, (char)ch);
  }
}
