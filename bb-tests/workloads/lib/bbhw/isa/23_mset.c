#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"

#define BB_MSET_FUNC7 23

#define bb_mset(bank_id, alloc, row, col)                        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0,                        \
      (FIELD(bank_id, 0, 4) | FIELD(row, 5, 9) | FIELD(col, 10, 14) | FIELD(alloc, 15, 15)) ,          \
      BB_MSET_FUNC7)

#define bb_mem_release(bank_id) bb_mset(bank_id, 0, 0, 0);

#define bb_mem_alloc(bank_id, row, col) bb_mset(bank_id, 1, row, col)

#endif // _BB_MSET_H_
