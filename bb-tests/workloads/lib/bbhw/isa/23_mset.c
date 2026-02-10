#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"

#define BB_MSET_FUNC7 23

#define bb_mset(relase_en, bank_id, alloc_en, row, col)                        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(relase_en, 0, 0) | FIELD(bank_id, 1, 13)),                        \
      (FIELD(alloc_en, 0, 0) | FIELD(row, 1, 5) | FIELD(col, 6, 13)),          \
      BB_MSET_FUNC7)

#define bb_mem_release(bank_id) bb_mset(1, bank_id, 0, 0, 0);

#define bb_mem_alloc(bank_id, row, col) bb_mset(0, bank_id, 1, row, col)

#define bb_vbank_config(bank_id, is_acc, alloc)                                \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, FIELD(bank_id, 0, 4) | FIELD(is_acc, 5, 5) | FIELD(alloc, 6, 6),      \
      BB_MSET_FUNC7)
#endif // _BB_MSET_H_
