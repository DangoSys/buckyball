#ifndef _BB_SNN_H_
#define _BB_SNN_H_

#include "isa.h"

#define BB_SNN_FUNC7 41
#define bb_snn(op1_addr, wr_addr, iter, threshold, leak_factor)                \
  BUCKYBALL_INSTRUCTION_R_R(FIELD(op1_addr, 0, 14),                            \
                            (FIELD(wr_addr, 0, 14) | FIELD(iter, 15, 24) |     \
                             FIELD(threshold, 25, 32) |                        \
                             FIELD(leak_factor, 33, 40)),                      \
                            BB_SNN_FUNC7)

#endif // _BB_SNN_H_
