#ifndef _BB_GEMMINI_FLUSH_H_
#define _BB_GEMMINI_FLUSH_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define BB_GEMMINI_FLUSH_RS2 4ULL

// Flush the matrix array state
#define bb_gemmini_flush()                                                     \
  BUCKYBALL_INSTRUCTION_R_R(0, BB_GEMMINI_FLUSH_RS2, BB_FUNC7(GEMMINI_FLUSH))

#endif // _BB_GEMMINI_FLUSH_H_
