#ifndef _BB_GEMMINI_CONFIG_H_
#define _BB_GEMMINI_CONFIG_H_

#include "isa.h"

#define BB_GEMMINI_CONFIG_FUNC7 2

// Configure Gemmini systolic array
// All config parameters go in rs2 (special), starting from bit 4.
// rs2[3:0] is the sub-command field and remains 0 for this instruction.
// dataflow: 0=OS, 1=WS
// activation: 0=none, 1=relu
// a_transpose, b_transpose: transpose flags
// in_shift: right-shift amount for output
#define bb_gemmini_config(dataflow, activation, a_transpose, b_transpose,      \
                          in_shift)                                            \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(dataflow, 4, 4) | FIELD(activation, 5, 6) | \
                             FIELD(a_transpose, 7, 7) |                        \
                             FIELD(b_transpose, 8, 8) |                        \
                             FIELD(in_shift, 9, 40)),                          \
                            BB_GEMMINI_CONFIG_FUNC7)

#endif // _BB_GEMMINI_CONFIG_H_
