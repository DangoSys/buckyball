#ifndef _BB_MVOUT_H_
#define _BB_MVOUT_H_

#include "isa.h"
#include <dma.h>

#define BB_MVOUT_FUNC7 16

/* Linux needs private destination pages before a DMA store. */
#if defined(__linux__)
#define BB_MVOUT_TOUCH(addr, depth, stride, bank)                              \
  dma_touch_mvout((void *)(addr), (depth), (stride), (bank))
#else
#define BB_MVOUT_TOUCH(addr, depth, stride, bank)                              \
  do {                                                                         \
  } while (0)
#endif

#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  do {                                                                         \
    uintptr_t _bb_mo_addr = (uintptr_t)(mem_addr);                             \
    uint32_t _bb_mo_bank = (uint32_t)(bank_id);                                \
    uint64_t _bb_mo_depth = (uint64_t)(depth);                                 \
    uint64_t _bb_mo_stride = (uint64_t)(stride);                               \
    BB_MVOUT_TOUCH(_bb_mo_addr, _bb_mo_depth, _bb_mo_stride, _bb_mo_bank);     \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (BB_BANK0(_bb_mo_bank) | BB_ITER(_bb_mo_depth)),                       \
        (FIELD(_bb_mo_addr, 0, 38) | FIELD(_bb_mo_stride, 39, 57)),            \
        BB_MVOUT_FUNC7);                                                       \
  } while (0)

static inline void bb_mvout_group(uintptr_t address, uint32_t bank,
                                  uint32_t group, uint64_t depth,
                                  uint64_t stride) {
#if defined(__linux__)
  dma_touch_mvout_group((void *)address, depth, stride);
#endif
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank) | BB_ITER(depth),
                            FIELD(address, 0, 38) | FIELD(stride, 39, 57) |
                                FIELD(group, 58, 62) | (UINT64_C(1) << 63),
                            BB_MVOUT_FUNC7);
}

#endif // _BB_MVOUT_H_
