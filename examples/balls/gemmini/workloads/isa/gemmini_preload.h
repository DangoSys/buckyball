#ifndef _BB_GEMMINI_PRELOAD_H_
#define _BB_GEMMINI_PRELOAD_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define BB_GEMMINI_PRELOAD_RS2 1ULL

// Preload D/B matrix into matrix array
// op1_bank_id: source bank for D (OS) or B (WS)
// wr_bank_id: destination bank for C output
// iter: number of rows to preload
#define bb_gemmini_preload(op1_bank_id, wr_bank_id, iter)                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),          \
      BB_GEMMINI_PRELOAD_RS2, BB_FUNC7(GEMMINI_PRELOAD))

#endif // _BB_GEMMINI_PRELOAD_H_
