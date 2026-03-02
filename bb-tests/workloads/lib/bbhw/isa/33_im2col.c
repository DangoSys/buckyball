#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 33

#define bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, \
                  startcol)                                                    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_RD0 | BB_WR),         \
      (FIELD(kcol, 0, 3) | FIELD(krow, 4, 7) | FIELD(incol, 8, 12) |           \
       FIELD(inrow, 13, 22) | FIELD(startcol, 23, 27) |                        \
       FIELD(startrow, 28, 37)),                                               \
      BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
