#ifndef _BB_FENCE_H_
#define _BB_FENCE_H_

#include "isa.h"

#define BB_FENCE_FUNC7 0

static inline void bb_dma_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}
#define bb_fence()                                                             \
  do {                                                                         \
    BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_FENCE_FUNC7);                           \
    bb_dma_fence();                                                            \
  } while (0)

#endif // _BB_FENCE_H_
