#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 33

#define bb_im2col(op1_bank_id, wr_bank_id, krow, kcol, inrow, incol, startrow, \
                  startcol)                                                    \
    BUCKYBALL_INSTRUCTION_R_R(                                                 \
        (FIELD(op1_bank_id, 0, 7) | FIELD(wr_bank_id, 8, 15)),                 \
        (FIELD(kcol, 16, 19) | FIELD(krow, 20, 23) | FIELD(incol, 24, 28) |    \
         FIELD(inrow, 29, 38) | FIELD(startcol, 39, 43) |                      \
         FIELD(startrow, 44, 53)),                                             \
        BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
