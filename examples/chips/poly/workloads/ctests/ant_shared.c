#include "ant_shared.h"

uint64_t ant_main(struct ant_shared_args *args) {
  volatile uint64_t *shared = (volatile uint64_t *)args->shared;
  if (args->producer) {
    shared[0] = args->value;
    __asm__ volatile("fence rw, rw" ::: "memory");
    shared[1] = 1;
  } else {
    while (!shared[1]) {
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
    args->value = shared[0];
  }
  return args->value;
}
