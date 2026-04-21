#ifndef _BB_MXFP_H_
#define _BB_MXFP_H_

#include "isa.h"

#define BB_MXFP_FUNC7 55

// Basic version:
//   rs1 = bank0(read) | bank2(write) | iter
//   rs2 = 0
#define bb_mxfp(bank_id, wr_bank_id, iter)                                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)), 0,           \
      BB_MXFP_FUNC7)

// Extended version:
//   rs2 carries user-defined special field.
//   Useful later for format select / rounding mode / debug flags.
#define bb_mxfp_ex(bank_id, wr_bank_id, iter, special)                         \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)), (special),   \
      BB_MXFP_FUNC7)

#endif // _BB_MXFP_H_
