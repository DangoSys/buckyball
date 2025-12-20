#ifndef _BB_ABFT_SYSTOLIC_H_
#define _BB_ABFT_SYSTOLIC_H_

#include "isa.h"

#define BB_ABFT_SYSTOLIC_FUNC7 42
#define bb_abft_systolic(op1_addr, op2_addr, wr_addr, iter)                    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_addr, 0, 14) | FIELD(op2_addr, 15, 29)),                      \
      (FIELD(wr_addr, 0, 14) | FIELD(iter, 15, 24)), BB_ABFT_SYSTOLIC_FUNC7)

#endif // _BB_ABFT_SYSTOLIC_H_
