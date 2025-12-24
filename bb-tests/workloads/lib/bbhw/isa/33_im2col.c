#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 33

#define bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, \
                  startcol)                                                    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(wr_bank_id, 8, 15)),                   \
      (FIELD(kcol, 0, 3) | FIELD(krow, 4, 7) | FIELD(incol, 8, 12) |      \
       FIELD(inrow, 13, 17) | FIELD(startcol, 23, 27) |                        \
       FIELD(startrow, 28, 32)),                                               \
      BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
