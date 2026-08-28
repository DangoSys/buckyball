#ifndef _BB_TRANSPOSE_H_
#define _BB_TRANSPOSE_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// bb_transpose(op1_bank_id, wr_bank_id, iter, elem_bits)
// rs1 = banks | iter
// rs2[7:0] = elem_bits; rs2[63:8] = 0
#define bb_transpose(op1_bank_id, wr_bank_id, iter, elem_bits)                 \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),          \
      (FIELD((uint64_t)(elem_bits), 0, 7)), BB_FUNC7(TRANSPOSE))

#endif // _BB_TRANSPOSE_H_
