#ifndef BB_MXINT8UNPACK_H
#define BB_MXINT8UNPACK_H
#include <bb_isa.h>
static inline void bb_mxint8unpack(uint32_t input, uint32_t output,
                                   uint32_t blocks) {
  // Bank1 is not an operand and is emitted as zero.
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) |
                                BB_ITER(blocks),
                            0, FUNCT7(mxint8unpack, MXINT8UNPACK));
}
#endif
