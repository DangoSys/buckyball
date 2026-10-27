#ifndef _BB_VECMAT16_H_
#define _BB_VECMAT16_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// C += A^T * B with int8 inputs and int32 accumulators.
//   op1 bank (one group): A^T, iter rows of sixteen int8; row k is column k
//                         of the 16 x iter matrix A.
//   op2 bank (one group): B, iter rows of sixteen int8.
//   wr bank (four groups): C, sixteen rows of sixteen int32; group g holds
//                          columns 4g..4g+3 of each row. The old C is read
//                          and accumulated, so rs1[29] is always set.
//   iter: the reduction depth K, 1..bank depth. rs2 is reserved and zero.
#define bb_vecmat16(op1_bank_id, op2_bank_id, wr_bank_id, iter)                \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                             BB_BANK2(wr_bank_id) | BB_READ_WR_OLD |           \
                             BB_ITER(iter)),                                   \
                            0, FUNCT7(vector, VECMAT16))

#endif // _BB_VECMAT16_H_
