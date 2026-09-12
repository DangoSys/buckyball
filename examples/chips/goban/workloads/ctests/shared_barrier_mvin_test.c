#include "goban.h"
#include "scu.h"

#define DIM 16
#define NCORES BB_CORES_PER_TILE

static elem_t src[NCORES][DIM * DIM] __attribute__((aligned(128)));
static elem_t dst[NCORES][DIM * DIM] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

int main(void) {
  int hart = bb_get_hart_id();
  core_id_t id = bb_get_core_id();
  int cid = (int)id.core;
  elem_t pat = (elem_t)(cid + 1);

  for (int i = 0; i < DIM * DIM; i++) {
    src[cid][i] = pat;
    dst[cid][i] = 0;
  }

  int bank = bb_shared_bank(cid);
  bb_mem_alloc(bank, 1, 1);
  bb_tile_barrier();
  bb_mvin((uintptr_t)src[cid], bank, DIM, 1);
  bb_mvout((uintptr_t)dst[cid], bank, DIM, 1);
  bb_fence();

  int ok = 1;
  for (int i = 0; i < DIM * DIM; i++) {
    if (dst[cid][i] != pat) {
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;
  bb_tile_barrier();
  bb_mem_release(bank);
  bb_tile_barrier();

  int all_ok = 1;
  if (cid == 0) {
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i])
        all_ok = 0;
    }
    scu_puts(hart, all_ok ? "shared_barrier_mvin PASSED\n"
                          : "shared_barrier_mvin FAILED\n");
  }
  return cid == 0 && !all_ok;
}
