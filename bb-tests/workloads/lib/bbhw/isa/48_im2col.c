#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 48

#define bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, \
                  startcol, col_step)                                          \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id)),                          \
      (FIELD(kcol, 0, 7) | FIELD(krow, 8, 15) | FIELD(incol, 16, 23) |         \
       FIELD(inrow, 24, 31) | FIELD(startcol, 32, 39) |                        \
       FIELD(startrow, 40, 47) | FIELD(col_step, 48, 55)),                     \
      BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
