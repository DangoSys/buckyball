#ifndef _BB_MVOUT_H_
#define _BB_MVOUT_H_

#include "isa.h"
#include <bbhw/mem/mem.h>

#define BB_MVOUT_FUNC7 16

/* Linux needs private destination pages before a DMA store. */
#if defined(__linux__)
#define BB_MVOUT_TOUCH(addr, depth, stride, bank)                              \
  bb_dma_touch_mvout((void *)(addr), (depth), (stride), (bank))
#else
#define BB_MVOUT_TOUCH(addr, depth, stride, bank)                              \
  do {                                                                         \
  } while (0)
#endif

#if defined(BUCKYBALL_RUSHB)
#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  do {                                                                         \
    uintptr_t _bb_mo_addr = (uintptr_t)(mem_addr);                             \
    uint32_t _bb_mo_bank = (uint32_t)(bank_id);                                \
    uint64_t _bb_mo_depth = (uint64_t)(depth);                                 \
    uint64_t _bb_mo_stride = (uint64_t)(stride);                               \
    BB_MVOUT_TOUCH(_bb_mo_addr, _bb_mo_depth, _bb_mo_stride, _bb_mo_bank);     \
    bb_dma_cache_flush();                                                      \
    rushb_mvout(                                                               \
        BUCKYBALL_RUSHB_CORE,                                                  \
        (uint64_t)(BB_BANK0(_bb_mo_bank) | BB_ITER(_bb_mo_depth)),             \
        (uint64_t)(FIELD(_bb_mo_addr, 0, 38) | FIELD(_bb_mo_stride, 39, 57)),  \
        (void *)_bb_mo_addr);                                                  \
  } while (0)
#else
#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  do {                                                                         \
    uintptr_t _bb_mo_addr = (uintptr_t)(mem_addr);                             \
    uint32_t _bb_mo_bank = (uint32_t)(bank_id);                                \
    uint64_t _bb_mo_depth = (uint64_t)(depth);                                 \
    uint64_t _bb_mo_stride = (uint64_t)(stride);                               \
    BB_MVOUT_TOUCH(_bb_mo_addr, _bb_mo_depth, _bb_mo_stride, _bb_mo_bank);     \
    bb_dma_cache_flush();                                                      \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (BB_BANK0(_bb_mo_bank) | BB_ITER(_bb_mo_depth)),                       \
        (FIELD(_bb_mo_addr, 0, 38) | FIELD(_bb_mo_stride, 39, 57)),            \
        BB_MVOUT_FUNC7);                                                       \
  } while (0)
#endif

#endif // _BB_MVOUT_H_
