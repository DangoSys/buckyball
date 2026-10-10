#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"
#include <dma.h>

#define BB_MSET_FUNC7 32

#define BB_MSET_CLEAR_BIT 11

#define BB_MSET_RS2(row, col, alloc, clear)                                    \
  (FIELD(row, 0, 4) | FIELD(col, 5, 9) | FIELD(alloc, 10, 10) |                \
   FIELD(clear, BB_MSET_CLEAR_BIT, BB_MSET_CLEAR_BIT))

static inline void bb_mset(uint32_t bank_id, uint32_t alloc, uint32_t row,
                           uint32_t col) {
#if defined(__linux__)
  if (alloc)
    dma_bank_allocate(bank_id, col);
  else
    dma_bank_set_cols(bank_id, 0);
#endif
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank_id), BB_MSET_RS2(row, col, alloc, 0),
                            BB_MSET_FUNC7);
}

static inline void bb_mset_clear(uint32_t bank_id, uint32_t row, uint32_t col) {
#if defined(__linux__)
  dma_bank_allocate(bank_id, col);
#endif
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank_id), BB_MSET_RS2(row, col, 1, 1),
                            BB_MSET_FUNC7);
}

static inline void bb_mem_transfer(uint32_t source, uint32_t target) {
#if defined(__linux__)
  dma_bank_transfer(source, target);
#endif
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(source) | BB_BANK2(target),
                            UINT64_C(1) << 12, BB_MSET_FUNC7);
}

#define bb_mem_release(bank_id) bb_mset((bank_id), 0, 0, 0)
#define bb_mem_alloc(bank_id, row, col) bb_mset((bank_id), 1, (row), (col))

#endif // _BB_MSET_H_
