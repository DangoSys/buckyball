#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Square-window im2col (iter != 0):
//   xs2[7:0]   ksize
//   xs2[15:8]  stride
//   xs2[23:16] padding   (symmetric zero-pad on every edge)
//   xs2[31:24] start_col (skip into padded space; asym pad co-design)
//   xs2[39:32] start_row
//
// outputDim = floor((iter + 2*padding - ksize - start) / stride) + 1
// Asym pad (lo, hi): padding=max(lo,hi), start=padding-lo.
#define bb_im2col(op1_bank_id, wr_bank_id, iter, ksize, stride, padding)       \
  bb_im2col_ex(op1_bank_id, wr_bank_id, iter, ksize, stride, padding, 0, 0)

#define bb_im2col_ex(op1_bank_id, wr_bank_id, iter, ksize, stride, padding,    \
                     start_row, start_col)                                     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),          \
      (FIELD(ksize, 0, 7) | FIELD(stride, 8, 15) | FIELD(padding, 16, 23) |    \
       FIELD(start_col, 24, 31) | FIELD(start_row, 32, 39)),                   \
      BB_FUNC7(IM2COL))

#endif // _BB_IM2COL_H_
