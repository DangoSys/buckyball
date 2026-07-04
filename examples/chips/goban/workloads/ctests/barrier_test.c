/*
 * barrier_test.c — Goban multi-core barrier smoke test.
 *
 * All nCores harts run this program concurrently (SPMD).
 * Test plan:
 *   1. Each core records a "before" cycle count.
 *   2. All cores execute bb_barrier() — hardware stalls until everyone arrives.
 *   3. Each core records an "after" cycle count.
 *   4. Core 0 prints a summary; all cores reach the final printf.
 *
 * Correctness criterion: if the simulation does not hang and all cores print
 * their completion message, the barrier works.
 */

#include "goban.h"
#include <stdio.h>

/* Shared flag written by each core after the barrier to verify ordering. */
static volatile int arrived[4] = {0, 0, 0, 0};

int main(void) {
  int cid = bb_get_core_id();

  printf("[core %d] starting\n", cid);

  /* --- Phase 1: mark arrival before barrier --- */
  arrived[cid] = 1;

  /* ============ BARRIER 1 ============ */
  bb_barrier();

  /* --- Phase 2: all cores should see every arrival flag set --- */
  int ok = 1;
  for (int i = 0; i < 4; i++) {
    if (!arrived[i]) {
      printf("[core %d] ERROR: arrived[%d] not set after barrier!\n", cid, i);
      ok = 0;
    }
  }

  if (ok) {
    printf("[core %d] after barrier: all arrival flags set — PASSED\n", cid);
  }

  /* ============ BARRIER 2 ============ */
  /* Verify barrier can be used more than once in the same program. */
  arrived[cid] = 2;
  bb_barrier();

  for (int i = 0; i < 4; i++) {
    if (arrived[i] != 2) {
      printf("[core %d] ERROR: arrived[%d] != 2 after barrier 2!\n", cid, i);
      ok = 0;
    }
  }

  if (ok) {
    printf("[core %d] barrier 2 PASSED\n", cid);
  }

  if (cid == 0) {
    printf("=== barrier_test %s ===\n", ok ? "PASSED" : "FAILED");
  }

  return ok ? 0 : 1;
}
