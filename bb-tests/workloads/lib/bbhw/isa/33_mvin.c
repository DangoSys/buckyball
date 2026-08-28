#ifndef _BB_MVIN_H_
#define _BB_MVIN_H_

#include "isa.h"

#define BB_MVIN_FUNC7 33

#if defined(BUCKYBALL_RUSHB)
#define bb_mvin(mem_addr, bank_id, depth, stride)                              \
  do {                                                                         \
    bb_dma_cache_flush();                                                      \
    rushb_mvin((uint64_t)(BB_BANK0(bank_id) | BB_ITER(depth)),                 \
               (uint64_t)(FIELD(mem_addr, 0, 38) | FIELD(stride, 39, 57)),     \
               (const void *)(uintptr_t)(mem_addr));                           \
  } while (0)
#else
#define bb_mvin(mem_addr, bank_id, depth, stride)                              \
  do {                                                                         \
    bb_dma_cache_flush();                                                      \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (BB_BANK0(bank_id) | BB_ITER(depth)),                                  \
        (FIELD(mem_addr, 0, 38) | FIELD(stride, 39, 57)), BB_MVIN_FUNC7);      \
  } while (0)
#endif

#endif // _BB_MVIN_H_
