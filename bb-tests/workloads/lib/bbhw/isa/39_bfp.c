#ifndef _BB_BFP_H_
#define _BB_BFP_H_

#include "isa.h"

#define BB_BFP_FUNC7 39

#define bb_BFP(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)        \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                             BB_BANK2(wr_bank_id) | BB_RD0 | BB_RD1 | BB_WR),  \
                            (FIELD(iter, 0, 9) | FIELD(mode, 10, 63)),         \
                            BB_BFP_FUNC7)

#endif // _BB_BFP_H_