#ifndef _BB_CIM_H_
#define _BB_CIM_H_

#include "isa.h"

#define BB_CIM_FUNC7 44
#define bb_cim(op1_bank_id, op2_bank_id, result_bank_id, iter, rows, cols, op_type)     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(op2_bank_id, 8, 15)),                      \
      (FIELD(result_bank_id, 0, 7) | FIELD(iter, 8, 23) | FIELD(rows, 24, 39) | \
       FIELD(cols, 40, 55) | FIELD(op_type, 56, 59)),                          \
      BB_CIM_FUNC7)

#endif // _BB_CIM_H_
