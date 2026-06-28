/*
 * barrier_mvin_test.c — Goban multi-core parallel mvin/mvout + barrier test.
 *
 * SPMD: all 4 cores run this program simultaneously.
 *
 * Each core:
 *   1. Allocates its own shared bank (bank = bb_shared_bank(cid)).
 *   2. Fills the bank with a core-specific pattern (mvin).
 *   3. Reads it back (mvout) and verifies locally.
 *   4. Calls bb_barrier() — waits for all cores.
 *   5. Records a "done" flag in shared memory.
 *   6. Core 0 prints the summary.
 *
 * This test exercises:
 *   - Per-core shared-bank mvin/mvout
 *   - bb_barrier() synchronization across all 4 cores
 *   - Parallel hardware accelerator utilization
 */

#include "goban.h"
#include "scu.h"
#include <string.h>

static void log_core_msg(int hart, int cid, const char *msg) {
  scu_puts(hart, "[core ");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, "] ");
  scu_puts(hart, msg);
  scu_putc(hart, '\n');
}


static void log_mismatch(int hart, int cid, int idx, elem_t got, elem_t expected) {
  scu_puts(hart, "[core ");
  scu_put_u32(hart, (uint32_t)cid);
  scu_puts(hart, "] ERROR: mvout mismatch idx=");
  scu_put_u32(hart, (uint32_t)idx);
  scu_puts(hart, " got=");
  scu_put_u32(hart, (uint32_t)got);
  scu_puts(hart, " expected=");
  scu_put_u32(hart, (uint32_t)expected);
  scu_putc(hart, '\n');
}

static void log_summary(int hart, const char *status) {
  scu_puts(hart, "=== barrier_mvin_test ");
  scu_puts(hart, status);
  scu_puts(hart, " ===\n");
}

#define DIM    16
#define NCORES 4

/* Per-core scratchpad buffers — compiler places these in BSS (all harts share
   the same virtual address space, but each core writes its own slot). */
static elem_t  src[NCORES][DIM * DIM] __attribute__((aligned(128)));
static elem_t  dst[NCORES][DIM * DIM] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

int main(void) {
  int hart = bb_get_hart_id();
  int cid = bb_get_core_id();

  log_core_msg(hart, cid, "starting mvin/mvout");

  /* ---- Step 1: fill src with a core-specific pattern ---- */
  elem_t pat = (elem_t)(cid + 1);   /* core 0 → 1, core 1 → 2, … */
  for (int i = 0; i < DIM * DIM; i++) {
    src[cid][i] = pat;
  }

  /* ---- Step 2: mvin -> shared bank <cid> ---- */
  int bank = bb_shared_bank(cid);   /* force the decoder onto the shared path */
  bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)src[cid], bank, DIM, 1);

  /* ---- Step 3: mvout → dst ---- */
  memset(dst[cid], 0, sizeof(dst[cid]));
  bb_mvout((uintptr_t)dst[cid], bank, DIM, 1);
  bb_fence();
  bb_mem_release(bank);

  /* ---- Step 4: verify locally ---- */
  int ok = 1;
  for (int i = 0; i < DIM * DIM; i++) {
    if (dst[cid][i] != pat) {
      log_mismatch(hart, cid, i, dst[cid][i], pat);
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;
  log_core_msg(hart, cid, ok ? "mvin/mvout PASSED" : "mvin/mvout FAILED");

  /* ============ BARRIER: wait for all cores to finish ============ */
  bb_barrier();

  /* ---- Step 5: core 0 collects results ---- */
  if (cid == 0) {
    int all_ok = 1;
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i]) {
        all_ok = 0;
        scu_puts(hart, "[core 0] a core reported FAILED\n");
      }
    }
    log_summary(hart, all_ok ? "PASSED" : "FAILED");
  }

  return core_ok[cid] ? 0 : 1;
}
