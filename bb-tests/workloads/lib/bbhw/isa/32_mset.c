#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"
#include <bbhw/mem/mem.h>

#define BB_MSET_FUNC7 32

#define BB_MSET_CLEAR_BIT 11

#define BB_MSET_RS2(row, col, alloc, clear)                                    \
  (FIELD(row, 0, 4) | FIELD(col, 5, 9) | FIELD(alloc, 10, 10) |                \
   FIELD(clear, BB_MSET_CLEAR_BIT, BB_MSET_CLEAR_BIT))

#if defined(BUCKYBALL_RUSHB)
#define bb_mset(bank_id, alloc, row, col)                                      \
  rushb_mset((uint64_t)BB_BANK0(bank_id),                                      \
             (uint64_t)BB_MSET_RS2(row, col, alloc, 0))
#define bb_mset_clear(bank_id, row, col)                                       \
  rushb_mset((uint64_t)BB_BANK0(bank_id), (uint64_t)BB_MSET_RS2(row, col, 1, 1))
#else
#define bb_mset(bank_id, alloc, row, col)                                      \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank_id),                                 \
                            BB_MSET_RS2(row, col, alloc, 0), BB_MSET_FUNC7)
#define bb_mset_clear(bank_id, row, col)                                       \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank_id), BB_MSET_RS2(row, col, 1, 1),    \
                            BB_MSET_FUNC7)
#endif

#define bb_mem_release(bank_id)                                                \
  do {                                                                         \
    uint32_t _bb_mr_bank = (uint32_t)(bank_id);                                \
    bb_dma_bank_set_cols(_bb_mr_bank, 1);                                      \
    bb_mset(_bb_mr_bank, 0, 0, 0);                                             \
  } while (0)

#define bb_mem_alloc(bank_id, row, col)                                        \
  do {                                                                         \
    uint32_t _bb_ma_bank = (uint32_t)(bank_id);                                \
    uint32_t _bb_ma_col = (uint32_t)(col);                                     \
    bb_dma_bank_set_cols(_bb_ma_bank, _bb_ma_col);                             \
    bb_mset(_bb_ma_bank, 1, (row), _bb_ma_col);                                \
  } while (0)

#endif // _BB_MSET_H_
