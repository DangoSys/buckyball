#include "ant_parallel.h"
#include <bbhw/isa/isa.h>

uint64_t ant_main(const struct ant_parallel_args *args) {
  volatile uint64_t *ready = (volatile uint64_t *)args->shared;
  if (args->publish)
    *ready = 1;
  else
    while (!*ready) {
    }
  __asm__ volatile("fence rw, rw" ::: "memory");
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->input, 3, 1, 1);
  bb_mvout(args->output, 3, 1, 1);
  bb_fence();
  bb_mset(3, 0, 0, 0);
  return 0;
}
