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
  return cid == 0 ? tile : HIDDEN_HART_BASE + tile * (NCORES - 1) + (cid - 1);
}

static void wait_all_ready(int hart) {
  scu_set_ready(hart, 1);
  if (hart == 0) {
    int ready = 0;
    while (!ready) {
      ready = 1;
      for (int i = 0; i < NTILES * NCORES; ++i) {
        if (scu_get_ready(i) != 1)
          ready = 0;
      }
      scu_poll_pause();
    }
    for (int i = 0; i < NTILES * NCORES; ++i)
      scu_set_ready(i, 2);
    return;
  }
  while (scu_get_ready(hart) != 2)
    scu_poll_pause();
}

int main(void) {
  int hart = bb_get_hart_id();
  int tile = bb_get_tile_id();
  int cid = bb_get_tile_core_id();

  if (hart < 0 || hart >= NTILES * NCORES || tile < 0 || tile >= NTILES ||
      cid < 0 || cid >= NCORES || hart != expected_hart_id(tile, cid)) {
    while (1)
      asm volatile("wfi");
  }

  wait_all_ready(hart);
  while (1)
    scu_putc(hart, (char)scu_getc(hart));
}
