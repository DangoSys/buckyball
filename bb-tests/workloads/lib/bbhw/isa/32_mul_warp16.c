#ifndef _BB_MUL_WARP16_H_
#define _BB_MUL_WARP16_H_

#include "isa.h"

#define BB_MUL_WARP16_FUNC7 32

#define bb_mul_warp16(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                  \
      (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23) | FIELD(mode, 24, 24)),    \
      BB_MUL_WARP16_FUNC7)

#endif // _BB_MUL_WARP16_H_
