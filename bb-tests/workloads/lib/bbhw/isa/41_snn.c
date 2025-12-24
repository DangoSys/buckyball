#ifndef _BB_SNN_H_
#define _BB_SNN_H_

#include "isa.h"

#define BB_SNN_FUNC7 41
#define bb_snn(op1_bank_id, wr_bank_id, iter, threshold, leak_factor)                \
  BUCKYBALL_INSTRUCTION_R_R(FIELD(op1_bank_id, 0, 7),                            \
                            (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23) |     \
                             FIELD(threshold, 24, 31) |                        \
                             FIELD(leak_factor, 32, 39)),                      \
                            BB_SNN_FUNC7)

#endif // _BB_SNN_H_
