#ifndef _BB_GEMMINI_FLUSH_H_
#define _BB_GEMMINI_FLUSH_H_

#include "isa.h"

#define BB_GEMMINI_FLUSH_FUNC7 46

// Flush the systolic array state
#define bb_gemmini_flush()                                                     \
  BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_GEMMINI_FLUSH_FUNC7)

#endif // _BB_GEMMINI_FLUSH_H_
