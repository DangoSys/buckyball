#include "ant_mesh.h"
#include <bbhw/isa/isa.h>
#include <params.h>

uint64_t entry(const struct ant_mesh_args *args) {
  if (args->index == 0) {
    bb_mem_alloc(0, 1, 1);
    bb_mvin(args->input, 0, 2, 1);
  }
  if (args->index == 1) {
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(4, 1, 1);
  }
  if (args->index == 2)
    bb_mem_alloc(3, 1, 1);
  if (args->index == 3)
    bb_mem_alloc(2, 1, 1);
  if (args->index == 4)
    bb_mem_alloc(1, 1, 1);

  /* All five Buckyball threads join each barrier. Each barrier drains the ROB.
   */
  bb_barrier();
  if (args->index == 0) {
    bb_mvover(0, 0, 1, 1, 1, 1, 1);
    bb_mvover(0, 0, 1, 3, 2, 1, 1);
  }
  bb_barrier();

  if (args->index == 1)
    bb_mvout(args->output + 0, 1, 2, 1);
  if (args->index == 3)
    bb_mvout(args->output + 32, 2, 2, 1);
  bb_barrier();

  if (args->index == 3)
    bb_mvover(3, 2, 1, 2, 3, 1, 1);
  bb_barrier();
  if (args->index == 2)
    bb_mvout(args->output + 64, 3, 2, 1);
  bb_barrier();

  if (args->index == 0)
    bb_mvover(0, 0, 1, 4, 1, 1, 1);
  if (args->index == 3)
    bb_mvover(3, 2, 1, 1, 4, 1, 1);
  bb_barrier();
  if (args->index == 4)
    bb_mvout(args->output + 96, 1, 2, 1);
  if (args->index == 1)
    bb_mvout(args->output + 128, 4, 2, 1);
  bb_barrier();

  return 0;
}
