#ifndef _MATMUL_WS_H_
#define _MATMUL_WS_H_

#include "isa.h"

#define BB_MATMUL_WS_FUNC7 27

#define bb_matmul_ws(op1_bank_id, op2_bank_id, wr_bank_id, iter)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                  \
      (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23) | FIELD(ws_flag, 24, 24)),    \
      BB_MATMUL_WS_FUNC7)

#endif // _MATMUL_WS_H_
