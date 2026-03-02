#ifndef _BB_RELU_H_
#define _BB_RELU_H_

#include "isa.h"

#define BB_RELU_FUNC7 38

#define bb_relu(bank_id, wr_bank_id, iter)                                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_RD0 | BB_WR),             \
      (FIELD(iter, 0, 9)), BB_RELU_FUNC7)

#endif // _BB_RELU_H_
