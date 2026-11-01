#ifndef _BB_MXQUANT_H_
#define _BB_MXQUANT_H_
#include <bb_isa.h>

#define bb_mxquant(input, output, count)                                       \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) |               \
                                BB_ITER(count),                                \
                            0, FUNCT7(mxquant, MXQUANT))
#endif
