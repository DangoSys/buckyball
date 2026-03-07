#ifndef _BB_GEMMINI_CONFIG_H_
#define _BB_GEMMINI_CONFIG_H_

#include "isa.h"

#define BB_GEMMINI_CONFIG_FUNC7 42

// Configure Gemmini systolic array
// dataflow: 0=OS, 1=WS
// activation: 0=none, 1=relu
// a_transpose, b_transpose: transpose flags
// in_shift: right-shift amount for output
#define bb_gemmini_config(dataflow, activation, a_transpose, b_transpose,      \
                          in_shift)                                            \
  BUCKYBALL_INSTRUCTION_R_R((FIELD(dataflow, 2, 2) | FIELD(activation, 3, 4) | \
                             FIELD(a_transpose, 8, 8) |                        \
                             FIELD(b_transpose, 9, 9)),                        \
                            (FIELD(in_shift, 0, 31)), BB_GEMMINI_CONFIG_FUNC7)

#endif // _BB_GEMMINI_CONFIG_H_
