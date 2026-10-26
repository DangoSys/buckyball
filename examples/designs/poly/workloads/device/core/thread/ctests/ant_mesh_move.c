#include "ant_mesh_move.h"
#include <bbhw/isa/isa.h>

enum { ROWS = 2, SOURCE_BANK = 1, TARGET_BANK = 8 };

uint64_t entry(const struct ant_mesh_move_args *args) {
  volatile uint64_t *sync = (volatile uint64_t *)args->sync;
  if (args->role == MESH_FILL) {
    bb_mset(SOURCE_BANK, 1, 1, 1);
    bb_mvin(args->input, SOURCE_BANK, ROWS, 1);
  } else if (args->role == MESH_MOVE) {
    while (!sync[0]) {
    }
    bb_mvover(0, SOURCE_BANK, 0, 1, TARGET_BANK, 0, ROWS);
    __asm__ volatile("fence rw, rw" ::: "memory");
    sync[1] = 1;
    bb_mset(SOURCE_BANK, 0, 0, 0);
  } else {
    bb_mset(TARGET_BANK, 1, 1, 1);
    sync[0] = 1;
    __asm__ volatile("fence rw, rw" ::: "memory");
    while (!sync[1]) {
    }
    bb_mvout(args->output, TARGET_BANK, ROWS, 1);
    bb_mset(TARGET_BANK, 0, 0, 0);
  }
  return 0;
}
