#ifndef _BB_DEQUANT_H_
#define _BB_DEQUANT_H_

#include "isa.h"

#define BB_DEQUANT_FUNC7 41

// bb_dequant(bank_id, wr_bank_id, iter, scale_fp32)
// scale_fp32 is a 32-bit FP32 value passed as uint32_t bit pattern
// Encoding: rs1 = BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_RD0 | BB_WR
//           rs2 = FIELD(iter, 0, 9) | FIELD(scale_fp32, 10, 41)
#define bb_dequant(bank_id, wr_bank_id, iter, scale_fp32)                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_RD0 | BB_WR),             \
      (FIELD(iter, 0, 9) | FIELD((uint64_t)(scale_fp32), 10, 41)),             \
      BB_DEQUANT_FUNC7)

#endif // _BB_DEQUANT_H_
