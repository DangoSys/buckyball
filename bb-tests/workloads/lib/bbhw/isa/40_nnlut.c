#ifndef _BB_NNLUT_H_
#define _BB_NNLUT_H_

#include "isa.h"

#define BB_NNLUT_FUNC7 40

#define bb_nnlut(op1_addr, wr_addr, iter)                                      \
  BUCKYBALL_INSTRUCTION_R_R(FIELD(op1_addr, 0, 14),                            \
                            (FIELD(wr_addr, 0, 14) | FIELD(iter, 15, 24)),     \
                            BB_NNLUT_FUNC7)

#endif // _BB_NNLUT_H_
