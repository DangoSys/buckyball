#ifndef _BB_TRANSFER_H_
#define _BB_TRANSFER_H_

#include "isa.h"

#define BB_TRANSFER_FUNC7 45

#define bb_transfer(op1_addr, wr_addr, iter)                                   \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      FIELD(op1_addr, 0, 14),                                                  \
      (FIELD(wr_addr, 0, 14) | FIELD((iter > 1023 ? 1023 : iter), 15, 24)),    \
      BB_TRANSFER_FUNC7)

#endif // _BB_TRANSFER_H_
