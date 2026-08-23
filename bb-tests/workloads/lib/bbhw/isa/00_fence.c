#ifndef _BB_FENCE_H_
#define _BB_FENCE_H_

#include "isa.h"

#define BB_FENCE_FUNC7 0

#define bb_dma_cache_flush() asm volatile("fence.i" ::: "memory")

#define bb_fence()                                                             \
  do {                                                                         \
    BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_FENCE_FUNC7);                           \
    asm volatile("fence rw, rw" ::: "memory");                                 \
    bb_dma_cache_flush();                                                      \
  } while (0)

#endif // _BB_FENCE_H_
