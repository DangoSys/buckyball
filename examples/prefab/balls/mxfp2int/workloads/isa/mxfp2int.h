#ifndef _BB_MXFP2INT_H_
#define _BB_MXFP2INT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Dequantizes iter MXFP4 blocks of 32 elements to int8.
//   op1 bank: block b in row b, element e in byte e / 2, low nibble first;
//             the element value is raw - 8.
//   op2 bank: E8M0 scale of block b in byte b % 16 of row b / 16.
//   wr bank:  elements 0..15 of block b in row 2b, 16..31 in row 2b + 1,
//             saturate(value * 2^(scale - 127)) with the right shift
//             rounding toward minus infinity.
//   iter: the number of blocks, 1..half the bank depth. rs2 is reserved and
//   zero.
#define bb_mxfp2int(bank_id, scale_bank_id, wr_bank_id, iter)                  \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_BANK1(scale_bank_id) |     \
                             BB_BANK2(wr_bank_id) | BB_ITER(iter)),            \
                            0, FUNCT7(mxfp2int, MXFP2INT))

#endif // _BB_MXFP2INT_H_
