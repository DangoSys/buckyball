#include "copy.h"
#include <bbhw/isa/isa.h>

uint64_t entry(const struct copy_task *args) {
  bb_mem_alloc(0, 1, 1);
  bb_mvin(args->input, 0, args->rows, 1);
  bb_mvout(args->output, 0, args->rows, 1);
  bb_mem_release(0);
  return 0;
}
