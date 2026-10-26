#ifndef _BB_MATADD_H_
#define _BB_MATADD_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_matadd(a_bank, b_bank, c_bank, iter)                                \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(a_bank) | BB_BANK1(b_bank) |             \
                             BB_BANK2(c_bank) | BB_ITER(iter)),                \
                            0, BB_FUNC7(MATADD))

#endif // _BB_MATADD_H_
