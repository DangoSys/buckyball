#include "goban.h"
#include "scu.h"

static void wait_all_ready(int hart) {
  scu_set_ready(hart, 1);
  if (hart == 0) {
    int ready = 0;
    while (!ready) {
      ready = 1;
      for (int i = 0; i < BB_TILE_NUM * BB_CORES_PER_TILE; ++i) {
        if (scu_get_ready(i) != 1)
          ready = 0;
      }
      scu_poll_pause();
    }
    for (int i = 0; i < BB_TILE_NUM * BB_CORES_PER_TILE; ++i)
      scu_set_ready(i, 2);
    return;
  }
  while (scu_get_ready(hart) != 2)
    scu_poll_pause();
}

int main(void) {
  int hart = bb_get_hart_id();
  core_id_t id = bb_get_core_id();
  int tile = (int)id.tile;
  int cid = (int)id.core;

  if (hart >= BB_TILE_NUM * BB_CORES_PER_TILE || tile >= BB_TILE_NUM ||
      cid >= BB_CORES_PER_TILE || hart != tile * BB_CORES_PER_TILE + cid) {
    while (1)
      asm volatile("wfi");
  }

  wait_all_ready(hart);
  while (1)
    scu_putc(hart, (char)scu_getc(hart));
}
