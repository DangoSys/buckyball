#include "goban.h"
#include "scu.h"

static volatile int arrived[BB_CORES_PER_TILE];
static volatile int core_ok[BB_CORES_PER_TILE];
static volatile int test_ok;

int main(void) {
  core_id_t id = bb_get_core_id();
  int cid = (int)id.core;

  arrived[cid] = 1;
  bb_tile_barrier();

  int ok = 1;
  for (int i = 0; i < BB_CORES_PER_TILE; i++) {
    if (!arrived[i]) {
      ok = 0;
    }
  }

  arrived[cid] = 2;
  bb_tile_barrier();

  for (int i = 0; i < BB_CORES_PER_TILE; i++) {
    if (arrived[i] != 2) {
      ok = 0;
    }
  }
  core_ok[cid] = ok;
  bb_tile_barrier();

  if (cid == 0) {
    test_ok = 1;
    for (int i = 0; i < BB_CORES_PER_TILE; i++) {
      if (!core_ok[i]) {
        test_ok = 0;
      }
    }
    scu_puts(0, "=== barrier_test ");
    scu_puts(0, test_ok ? "PASSED" : "FAILED");
    scu_puts(0, " ===\n");
  }

  bb_tile_barrier();
  return test_ok ? 0 : 1;
}
