#ifndef _BB_MATADD_H_
#define _BB_MATADD_H_

#include <bb_isa.h>

#define bb_matadd(a_bank, b_bank, c_bank, iter)                                \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(a_bank) | BB_BANK1(b_bank) |             \
                             BB_BANK2(c_bank) | BB_ITER(iter)),                \
                            0, FUNCT7(matadd, MATADD))

#endif // _BB_MATADD_H_
