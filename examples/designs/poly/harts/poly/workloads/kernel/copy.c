#include "copy.h"
#include <bbhw/isa/isa.h>

uint64_t entry(const struct copy_task *args) {
  bb_mem_alloc(0, 1, 1);
  for (uint64_t offset = 0; offset < args->rows;) {
    uint64_t rows = offset ? 256 : 2;
    if (rows > args->rows - offset)
      rows = args->rows - offset;
    bb_mvin(args->input + offset * 16, 0, rows, 1);
    bb_mvout(args->output + offset * 16, 0, rows, 1);
    offset += rows;
  }
  bb_mem_release(0);
  return 0;
}
