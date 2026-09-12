#include "goban.h"
#include "scu.h"

static volatile int core_ok[BB_TILE_NUM][BB_CORES_PER_TILE];
static volatile int test_ok[BB_TILE_NUM];

int main(void) {
  core_id_t id = bb_get_core_id();
  int hart = bb_get_hart_id();
  int expected = (int)id.tile * BB_CORES_PER_TILE + (int)id.core;
  core_ok[id.tile][id.core] = hart == expected;
  bb_tile_barrier();

  if (id.core == 0) {
    test_ok[id.tile] = 1;
    for (int core = 0; core < BB_CORES_PER_TILE; ++core)
      if (!core_ok[id.tile][core])
        test_ok[id.tile] = 0;
    if (id.tile == 0) {
      scu_puts(0, "goban_topology_smoke ");
      scu_puts(0, test_ok[id.tile] ? "PASSED\n" : "FAILED\n");
    }
  }

  bb_tile_barrier();
  return id.tile == 0 && id.core == 0 && !test_ok[id.tile];
}
