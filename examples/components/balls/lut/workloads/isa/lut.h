#ifndef _BB_LUT_H_
#define _BB_LUT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_lut(input_bank, lut_bank, output_bank, iter)                        \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK1(lut_bank) |        \
                                BB_BANK2(output_bank) | BB_ITER(iter),         \
                            0, BB_FUNC7(LUT))

#endif // _BB_LUT_H_
