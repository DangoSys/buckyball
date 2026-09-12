/*
 * shared_multigroup_mvin_test.c - Goban shared multi-group mvin/mvout test.
 *
 * Each core allocates a two-group shared bank and verifies that group_count
 * query drives the loader/storer across both groups.
 */

#include "goban.h"
#include "scu.h"
#include <string.h>

#define ROWS 16
#define ROW_ELEMS 16
#define GROUPS (BB_SHARED_PHYSICAL_BANK_NUM / BB_CORES_PER_TILE)
#define NCORES BB_CORES_PER_TILE
#define ELEMS (ROWS * ROW_ELEMS * GROUPS)

_Static_assert(
    BB_SHARED_PHYSICAL_BANK_NUM >= BB_CORES_PER_TILE &&
        BB_SHARED_PHYSICAL_BANK_NUM % BB_CORES_PER_TILE == 0,
    "shared multigroup test requires at least one shared bank per core");

static elem_t src[NCORES][ELEMS] __attribute__((aligned(128)));
static elem_t dst[NCORES][ELEMS] __attribute__((aligned(128)));
static volatile int core_ok[NCORES];

static volatile int test_ok;

int main(void) {
  core_id_t id = bb_get_core_id();
  int cid = (int)id.core;
  int bank = bb_shared_bank(cid);

  for (int i = 0; i < ELEMS; i++) {
    src[cid][i] = (elem_t)(cid * 11 + (i & 0x7f));
  }

  bb_mem_alloc(bank, 1, GROUPS);
  bb_tile_barrier();
  bb_mvin((uintptr_t)src[cid], bank, ROWS, 1);
  memset(dst[cid], 0, sizeof(dst[cid]));
  bb_mvout((uintptr_t)dst[cid], bank, ROWS, 1);
  bb_fence();

  int ok = 1;
  for (int i = 0; i < ELEMS; i++) {
    if (dst[cid][i] != src[cid][i]) {
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
    scu_puts(0, "=== shared_multigroup_mvin_test ");
    scu_puts(0, test_ok ? "PASSED" : "FAILED");
    scu_puts(0, " ===\n");
  }

  bb_tile_barrier();
  return test_ok ? 0 : 1;
}
