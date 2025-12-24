#ifndef _BB_TRANSFER_H_
#define _BB_TRANSFER_H_

#include "isa.h"

#define BB_TRANSFER_FUNC7 45

#define bb_transfer(op1_bank_id, wr_bank_id, iter)                                   \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      FIELD(op1_bank_id, 0, 7),                                                  \
      (FIELD(wr_bank_id, 0, 7) | FIELD((iter > 1023 ? 1023 : iter), 8, 23)),    \
      BB_TRANSFER_FUNC7)

#endif // _BB_TRANSFER_H_
