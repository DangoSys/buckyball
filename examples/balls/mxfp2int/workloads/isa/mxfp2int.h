#ifndef _BB_MXFP2INT_H_
#define _BB_MXFP2INT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Basic version:
//   rs1 = bank0(read) | bank2(write) | iter
//   rs2 = 0
#define bb_mxfp2int(bank_id, wr_bank_id, iter)                                 \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)), 0,           \
      BB_FUNC7(MXFP2INT))

// Extended version:
//   rs2 carries user-defined special field.
//   Useful later for format select / rounding mode / debug flags.
#define bb_mxfp2int_ex(bank_id, wr_bank_id, iter, special)                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)), (special),   \
      BB_FUNC7(MXFP2INT))

#endif // _BB_MXFP2INT_H_
