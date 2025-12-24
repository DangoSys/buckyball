#ifndef _BB_ABFT_SYSTOLIC_H_
#define _BB_ABFT_SYSTOLIC_H_

#include "isa.h"

#define BB_ABFT_SYSTOLIC_FUNC7 42
#define bb_abft_systolic(op1_bank_id, op2_bank_id, wr_bank_id, iter)                    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                      \
      (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23)), BB_ABFT_SYSTOLIC_FUNC7)

#endif // _BB_ABFT_SYSTOLIC_H_
