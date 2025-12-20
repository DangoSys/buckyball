#ifndef _BB_RELU_H_
#define _BB_RELU_H_

#include "isa.h"

#define BB_RELU_FUNC7 38

#define bb_relu(bank_id, wr_bank_id, iter)                                     \
  BUCKYBALL_INSTRUCTION_R_R(FIELD(bank_id, 0, 7),                              \
                            FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 17),      \
                            BB_RELU_FUNC7)

#endif // _BB_RELU_H_
