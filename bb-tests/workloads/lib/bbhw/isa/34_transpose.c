#ifndef _BB_TRANSPOSE_H_
#define _BB_TRANSPOSE_H_

#include "isa.h"

#define BB_TRANSPOSE_FUNC7 34

#define bb_transpose(op1_bank_id, wr_bank_id, iter, mode)                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(wr_bank_id, 8, 15)),                   \
      (FIELD(iter, 15, 24) | FIELD(mode, 25, 25)), BB_TRANSPOSE_FUNC7)

#endif // _BB_TRANSPOSE_H_
