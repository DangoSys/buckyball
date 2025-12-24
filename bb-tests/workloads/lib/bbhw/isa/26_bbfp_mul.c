#ifndef _BBFP_MUL_H_
#define _BBFP_MUL_H_

#include "isa.h"

#define BB_BBFP_MUL_FUNC7 26

#define bb_bbfp_mul(op1_bank_id, op2_bank_id, wr_bank_id, iter)                \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                  \
      (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23)), BB_BBFP_MUL_FUNC7)

#endif // _BBFP_MUL_H_
