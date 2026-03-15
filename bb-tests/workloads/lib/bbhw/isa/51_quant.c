#ifndef _BB_QUANT_H_
#define _BB_QUANT_H_

#include "isa.h"

#define BB_QUANT_FUNC7 51

// bb_quant(bank_id, wr_bank_id, iter, scale_fp32)
// scale_fp32 is a 32-bit FP32 value passed as uint32_t bit pattern
// Encoding: rs1 = banks | iter
//           rs2 = scale_fp32
#define bb_quant(bank_id, wr_bank_id, iter, scale_fp32)                        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),              \
      (FIELD((uint64_t)(scale_fp32), 0, 31)), BB_QUANT_FUNC7)

#endif // _BB_QUANT_H_
