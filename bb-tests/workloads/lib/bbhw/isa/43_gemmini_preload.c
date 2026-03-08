#ifndef _BB_GEMMINI_PRELOAD_H_
#define _BB_GEMMINI_PRELOAD_H_

#include "isa.h"

#define BB_GEMMINI_PRELOAD_FUNC7 43

// Preload D/B matrix into systolic array
// op1_bank_id: source bank for D (OS) or B (WS)
// wr_bank_id: destination bank for C output
// iter: number of rows to preload
#define bb_gemmini_preload(op1_bank_id, wr_bank_id, iter)                      \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) |    \
                             BB_RD0 | BB_WR | BB_ITER(iter)),                  \
                            0, BB_GEMMINI_PRELOAD_FUNC7)

#endif // _BB_GEMMINI_PRELOAD_H_
