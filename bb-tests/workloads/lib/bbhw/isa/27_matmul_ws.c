#ifndef _MATMUL_WS_H_
#define _MATMUL_WS_H_

#include "isa.h"

#define BB_MATMUL_WS_FUNC7 27

#define bb_matmul_ws(op1_bank_id, op2_bank_id, wr_bank_id, iter)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                  \
      (FIELD(wr_bank_id, 16, 23) | FIELD(iter, 24, 33) | FIELD(1, 34, 34)),    \
      BB_MATMUL_WS_FUNC7)

#endif // _MATMUL_WS_H_
