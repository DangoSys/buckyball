#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 33

#define bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, \
                  startcol)                                                    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(op1_bank_id, 0, 7) | FIELD(wr_bank_id, 8, 15)),                   \
      (FIELD(kcol, 26, 29) | FIELD(krow, 30, 33) | FIELD(incol, 34, 38) |      \
       FIELD(inrow, 39, 43) | FIELD(startcol, 49, 53) |                        \
       FIELD(startrow, 54, 58)),                                               \
      BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
