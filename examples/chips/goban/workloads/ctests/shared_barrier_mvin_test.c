#include "goban.h"
#include "scu.h"

#define DIM 16
#ifndef NCORES
#define NCORES 4
#endif

static elem_t src[NCORES][DIM * DIM] __attribute__((aligned(128)));
static elem_t dst[NCORES][DIM * DIM] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

int main(void) {
  int hart = bb_get_hart_id();
  int cid = bb_get_core_id();
  elem_t pat = (elem_t)(cid + 1);

  for (int i = 0; i < DIM * DIM; i++) {
    src[cid][i] = pat;
    dst[cid][i] = 0;
  }

  int bank = bb_shared_bank(cid);
  bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)src[cid], bank, DIM, 1);
  bb_mvout((uintptr_t)dst[cid], bank, DIM, 1);
  bb_fence();
  bb_mem_release(bank);

  int ok = 1;
  for (int i = 0; i < DIM * DIM; i++) {
    if (dst[cid][i] != pat) {
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;

  bb_barrier();

  if (cid == 0) {
    int all_ok = 1;
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i])
        all_ok = 0;
    }
    scu_puts(hart, all_ok ? "shared_barrier_mvin PASSED\n"
                          : "shared_barrier_mvin FAILED\n");
  }
  return core_ok[cid] ? 0 : 1;
}
