#ifndef GOBAN_H
#define GOBAN_H

#include <bbhw/isa/isa.h>
#include <multicore.h>
#include <stdint.h>

static inline int bb_shared_bank(int vbank_id) {
  return BB_SHARED_BANK_BASE + vbank_id;
}

static inline void bb_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline void bb_tile_barrier(void) {
  bb_cpu_fence();
  bb_barrier();
  bb_cpu_fence();
}

#endif // GOBAN_H
