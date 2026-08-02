#include "goban.h"
#include "scu.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16
#define NCORES 4
#ifndef NTILES
#define NTILES 64
#endif
#ifndef HIDDEN_HART_BASE
#define HIDDEN_HART_BASE 64
#endif

static elem_t src[DIM * DIM] __attribute__((aligned(128)));
static elem_t dst[DIM * DIM] __attribute__((aligned(128)));

static int expected_hart_id(int tile, int cid) {
  return cid == 0 ? tile : HIDDEN_HART_BASE + tile * (NCORES - 1) + (cid - 1);
}

int main(void) {
  int hart = bb_get_hart_id();
  int tile = bb_get_tile_id();
  int cid = bb_get_tile_core_id();

  if (hart < 0 || hart >= NTILES * NCORES || tile < 0 || tile >= NTILES ||
      cid < 0 || cid >= NCORES || hart != expected_hart_id(tile, cid)) {
    return 1;
  }

  scu_set_ready(hart, 1);
  if (hart != 0) {
    while (1)
      asm volatile("wfi");
  }

  for (uint32_t w = 0; w < 0x4000000U; w++) {
    int missing = 0;
    for (int h = 0; h < NTILES * NCORES; h++) {
      if (scu_get_ready(h) != 1) {
        missing = 1;
        break;
      }
    }
    if (!missing)
      break;
  }

  for (int i = 0; i < DIM * DIM; i++) {
    src[i] = 7;
    dst[i] = 0;
  }

  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)src, 0, DIM, 1);
  bb_mvout((uintptr_t)dst, 0, DIM, 1);
  bb_fence();
  bb_mem_release(0);

  int passed = 1;
  for (int i = 0; i < DIM * DIM; i++) {
    if (dst[i] != src[i])
      passed = 0;
  }
  printf("goban_hetero_t4c_smoke %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
