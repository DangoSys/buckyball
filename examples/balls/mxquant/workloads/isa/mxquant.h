#ifndef _BB_MXQUANT_H_
#define _BB_MXQUANT_H_
#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_mxquant(input, output, count)                                       \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) |               \
                                BB_ITER(count),                                \
                            0, BB_FUNC7(MXQUANT))
#endif
