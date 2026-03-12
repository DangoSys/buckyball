/*
 * barrier_mvin_test.c — Goban multi-core parallel mvin/mvout + barrier test.
 *
 * SPMD: all 4 cores run this program simultaneously.
 *
 * Each core:
 *   1. Allocates its own private bank (bank = cid).
 *   2. Fills the bank with a core-specific pattern (mvin).
 *   3. Reads it back (mvout) and verifies locally.
 *   4. Calls bb_barrier() — waits for all cores.
 *   5. Records a "done" flag in shared memory.
 *   6. Core 0 prints the summary.
 *
 * This test exercises:
 *   - Per-core private-bank mvin/mvout (no sharing needed)
 *   - bb_barrier() synchronization across all 4 cores
 *   - Parallel hardware accelerator utilization
 */

#include "goban.h"
#include <stdio.h>
#include <string.h>

#define DIM    16
#define NCORES 4

/* Per-core scratchpad buffers — compiler places these in BSS (all harts share
   the same virtual address space, but each core writes its own slot). */
static elem_t  src[NCORES][DIM * DIM] __attribute__((aligned(128)));
static elem_t  dst[NCORES][DIM * DIM] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

int main(void) {
  int cid = bb_get_core_id();

  printf("[core %d] starting mvin/mvout\n", cid);

  /* ---- Step 1: fill src with a core-specific pattern ---- */
  elem_t pat = (elem_t)(cid + 1);   /* core 0 → 1, core 1 → 2, … */
  for (int i = 0; i < DIM * DIM; i++) {
    src[cid][i] = pat;
  }

  /* ---- Step 2: mvin → shared bank <cid> ---- */
  int bank = cid;   /* each core uses its own bank, no conflict */
  bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)src[cid], bank, DIM, 1);

  /* ---- Step 3: mvout → dst ---- */
  memset(dst[cid], 0, sizeof(dst[cid]));
  bb_mvout((uintptr_t)dst[cid], bank, DIM, 1);
  bb_mem_release(bank);

  /* ---- Step 4: verify locally ---- */
  int ok = 1;
  for (int i = 0; i < DIM * DIM; i++) {
    if (dst[cid][i] != pat) {
      printf("[core %d] ERROR at [%d]: got %d, expected %d\n",
             cid, i, (int)dst[cid][i], (int)pat);
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;
  printf("[core %d] mvin/mvout %s\n", cid, ok ? "PASSED" : "FAILED");

  /* ============ BARRIER: wait for all cores to finish ============ */
  bb_barrier();

  /* ---- Step 5: core 0 collects results ---- */
  if (cid == 0) {
    int all_ok = 1;
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i]) {
        all_ok = 0;
        printf("[core 0] core %d reported FAILED\n", i);
      }
    }
    printf("=== barrier_mvin_test %s ===\n", all_ok ? "PASSED" : "FAILED");
  }

  return core_ok[cid] ? 0 : 1;
}
