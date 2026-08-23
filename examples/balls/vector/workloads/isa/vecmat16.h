#ifndef _BB_VECMAT16_H_
#define _BB_VECMAT16_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_vecmat16(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)          \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                             BB_BANK2(wr_bank_id) | BB_ITER(iter)),            \
                            (FIELD(mode, 0, 63)), BB_FUNC7(VECMAT16))

#endif // _BB_VECMAT16_H_
