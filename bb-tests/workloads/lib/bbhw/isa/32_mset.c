#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"

#define BB_MSET_FUNC7 32

#define bb_mset(bank_id, alloc, row, col)                                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(bank_id),                                                       \
      (FIELD(row, 0, 4) | FIELD(col, 5, 9) | FIELD(alloc, 10, 10)),            \
      BB_MSET_FUNC7)

#define bb_mem_release(bank_id) bb_mset(bank_id, 0, 0, 0);

#define bb_mem_alloc(bank_id, row, col) bb_mset(bank_id, 1, row, col)

#endif // _BB_MSET_H_
