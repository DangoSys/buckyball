#include "goban.h"
#include "scu.h"
#include <isa/transpose.h>

enum { DIM = 16 };
static int8_t input[BB_CORES_PER_TILE][DIM * DIM] __attribute__((aligned(128)));
static int8_t output[BB_CORES_PER_TILE][DIM * DIM] __attribute__((aligned(128)));
static volatile int core_ok[BB_CORES_PER_TILE];
static volatile int test_ok;

int main(void) {
  core_id_t id = bb_get_core_id();
  int core = (int)id.core;
  int shared = bb_shared_bank(core);
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col)
      input[core][row * DIM + col] = (int8_t)(core * 17 + row - col);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(shared, 1, 1);
  bb_tile_barrier();
  bb_mvin((uintptr_t)input[core], 0, DIM, 1);
  bb_transpose(0, shared, DIM, 8);
  bb_mvout((uintptr_t)output[core], shared, DIM, 1);
  bb_fence();

  int ok = 1;
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col)
      if (output[core][col * DIM + row] != input[core][row * DIM + col])
        ok = 0;
  core_ok[core] = ok;
  bb_tile_barrier();
  bb_mem_release(0);
  bb_mem_release(shared);
  bb_tile_barrier();

  if (core == 0) {
    test_ok = 1;
    for (int i = 0; i < BB_CORES_PER_TILE; ++i)
      if (!core_ok[i])
        test_ok = 0;
    scu_puts(0, test_ok ? "shared_transpose PASSED\n"
                        : "shared_transpose FAILED\n");
  }
  bb_tile_barrier();
  return test_ok ? 0 : 1;
}
