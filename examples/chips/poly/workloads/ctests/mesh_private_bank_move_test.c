#include <bbhw/isa/isa.h>
#include <multicore.h>
#include <params.h>
#include <topology.h>

#include <stdint.h>

#if BANK_WIDTH != 128
#error "Mesh private-Bank move requires 128-bit rows"
#endif
#if BB_SHARED_PHYSICAL_BANK_NUM < 1
#error "Mesh private-Bank move requires a Mesh SharedMem"
#endif

#define ROW_BYTES (BANK_WIDTH / 8)

static uint8_t source[2][ROW_BYTES] __attribute__((aligned(64)));
static uint8_t result[5][2][ROW_BYTES] __attribute__((aligned(64)));

int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.tile != 0)
    for (;;)
      asm volatile("wfi");

  if (id.core == 0) {
    for (int row = 0; row < 2; ++row)
      for (int i = 0; i < ROW_BYTES; ++i)
        source[row][i] = (uint8_t)((row ? 0xa0 : 0x30) + i);
    bb_mem_alloc(0, 1, 1);
    bb_mvin((uintptr_t)source, 0, 2, 1);
  }
  if (id.core == 1) {
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(4, 1, 1);
  }
  if (id.core == 2)
    bb_mem_alloc(3, 1, 1);
  if (id.core == 3)
    bb_mem_alloc(2, 1, 1);
  if (id.core == 4)
    bb_mem_alloc(1, 1, 1);

  /* All five Cores in Tile 0 join each barrier. Each barrier drains the ROB. */
  bb_barrier();
  if (id.core == 0) {
    bb_mesh_move(0, 0, 1, 1, 1, 1);
    bb_mesh_move(0, 0, 1, 3, 2, 1);
  }
  bb_barrier();

  if (id.core == 1)
    bb_mvout((uintptr_t)result[0], 1, 2, 1);
  if (id.core == 3)
    bb_mvout((uintptr_t)result[1], 2, 2, 1);
  bb_barrier();

  if (id.core == 3)
    bb_mesh_move(3, 2, 1, 2, 3, 1);
  bb_barrier();
  if (id.core == 2)
    bb_mvout((uintptr_t)result[2], 3, 2, 1);
  bb_barrier();

  if (id.core == 0)
    bb_mesh_move(0, 0, 1, 4, 1, 1);
  if (id.core == 3)
    bb_mesh_move(3, 2, 1, 1, 4, 1);
  bb_barrier();
  if (id.core == 4)
    bb_mvout((uintptr_t)result[3], 1, 2, 1);
  if (id.core == 1)
    bb_mvout((uintptr_t)result[4], 4, 2, 1);
  bb_barrier();

  if (id.core == 0) {
    for (int core = 0; core < 5; ++core)
      for (int i = 0; i < ROW_BYTES; ++i)
        if (result[core][1][i] != source[1][i])
          return 1;
    return 0;
  }
  for (;;)
    asm volatile("wfi");
}
