#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"

#define BB_MSET_FUNC7 23

#define bb_mset(relase_en, bank_id, alloc_en, bank_num, row, col)              \
  ({                                                                           \
    BUCKYBALL_INSTRUCTION_R_R((FIELD(alloc_en, 0, 0) | FIELD(bank_id, 1, 13)), \
                              (FIELD(relase_en, 0, 0) |                        \
                               FIELD(bank_num, 1, 5) | FIELD(row, 6, 9) |      \
                               FIELD(col, 10, 17)),                            \
                              BB_MSET_FUNC7);                                  \
    (bank_id);                                                                 \
  })

#endif // _BB_MSET_H_
