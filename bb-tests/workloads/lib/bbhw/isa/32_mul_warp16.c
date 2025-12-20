#ifndef _BB_MUL_WARP16_H_
#define _BB_MUL_WARP16_H_

#include "isa.h"

#define BB_MUL_WARP16_FUNC7 32

#define bb_mul_warp16(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                  \
      (FIELD(wr_bank_id, 16, 23) | FIELD(iter, 24, 33) | FIELD(mode, 34, 34)), \
      BB_MUL_WARP16_FUNC7)

#endif // _BB_MUL_WARP16_H_
