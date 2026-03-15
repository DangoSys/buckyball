#ifndef _BB_TRANSPOSE_H_
#define _BB_TRANSPOSE_H_

#include "isa.h"

#define BB_TRANSPOSE_FUNC7 49

#define bb_transpose(op1_bank_id, wr_bank_id, iter, mode)                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),          \
      (FIELD(mode, 0, 63)), BB_TRANSPOSE_FUNC7)

#endif // _BB_TRANSPOSE_H_
