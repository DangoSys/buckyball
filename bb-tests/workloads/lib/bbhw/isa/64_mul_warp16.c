#ifndef _BB_MUL_WARP16_H_
#define _BB_MUL_WARP16_H_

#include "isa.h"

#define BB_MUL_WARP16_FUNC7 64

#define bb_mul_warp16(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)        \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                             BB_BANK2(wr_bank_id) | BB_ITER(iter)),            \
                            (FIELD(mode, 0, 63)), BB_MUL_WARP16_FUNC7)

#endif // _BB_MUL_WARP16_H_
