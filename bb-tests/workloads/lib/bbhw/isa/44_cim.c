#ifndef _BB_CIM_H_
#define _BB_CIM_H_

#include "isa.h"

#define BB_CIM_FUNC7 44
#define bb_cim(op1_addr, op2_addr, result_addr, iter, rows, cols, op_type)     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_addr, 0, 14) | FIELD(op2_addr, 15, 29)),                      \
      (FIELD(result_addr, 0, 14) | FIELD(iter, 15, 24) | FIELD(rows, 25, 40) | \
       FIELD(cols, 41, 56) | FIELD(op_type, 57, 60)),                          \
      BB_CIM_FUNC7)

#endif // _BB_CIM_H_
