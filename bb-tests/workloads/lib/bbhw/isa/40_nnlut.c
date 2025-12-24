#ifndef _BB_NNLUT_H_
#define _BB_NNLUT_H_

#include "isa.h"

#define BB_NNLUT_FUNC7 40

#define bb_nnlut(op1_bank_id, wr_bank_id, iter)                                      \
  BUCKYBALL_INSTRUCTION_R_R(FIELD(op1_bank_id, 0, 7),                            \
                            (FIELD(wr_bank_id, 0, 7) | FIELD(iter, 8, 23)),     \
                            BB_NNLUT_FUNC7)

#endif // _BB_NNLUT_H_
