/*
 * shared_same_vbank_test.c - Goban shared query isolation test.
 *
 * All cores allocate the same shared vbank id. Correct behavior requires the
 * SharedMemBackend query and access paths to key mappings by (hart, vbank).
 */

#include "goban.h"
#include "scu.h"
#include <string.h>

#define ROWS 16
#define ROW_ELEMS 16
#define NCORES BB_CORES_PER_TILE

static elem_t src[NCORES][ROWS * ROW_ELEMS] __attribute__((aligned(128)));
static elem_t dst[NCORES][ROWS * ROW_ELEMS] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

static volatile int test_ok;

int main(void) {
  core_id_t id = bb_get_core_id();
  int cid = (int)id.core;
  int bank = VIRTUAL_BANK_NUM - 1;
  elem_t pat = (elem_t)(cid + 3);

  for (int i = 0; i < ROWS * ROW_ELEMS; i++) {
    src[cid][i] = pat;
  }

  bb_mem_alloc(bank, 1, 1);
  bb_tile_barrier();
  bb_mvin((uintptr_t)src[cid], bank, ROWS, 1);
  memset(dst[cid], 0, sizeof(dst[cid]));
  bb_mvout((uintptr_t)dst[cid], bank, ROWS, 1);
  bb_fence();

  int ok = 1;
  for (int i = 0; i < ROWS * ROW_ELEMS; i++) {
    if (dst[cid][i] != pat) {
      ok = 0;
      break;
    }
  }
  core_ok[cid] = ok;
  bb_tile_barrier();
  bb_mem_release(bank);
  bb_tile_barrier();

  if (cid == 0) {
    test_ok = 1;
    for (int i = 0; i < NCORES; i++) {
      if (!core_ok[i]) {
        test_ok = 0;
      }
    }
    scu_puts(0, "=== shared_same_vbank_test ");
    scu_puts(0, test_ok ? "PASSED" : "FAILED");
    scu_puts(0, " ===\n");
  }

  bb_tile_barrier();
  return test_ok ? 0 : 1;
}
