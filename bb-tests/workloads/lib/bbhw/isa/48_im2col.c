#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include "isa.h"

#define BB_IM2COL_FUNC7 48

// iter: input height and width (the input is an iter x iter square)
// ksize: square kernel height and width
// stride: row and column stride
// padding: zero-padding on every input edge
#define bb_im2col(op1_bank_id, wr_bank_id, iter, ksize, stride, padding)       \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),          \
      (FIELD(ksize, 0, 7) | FIELD(stride, 8, 15) | FIELD(padding, 16, 23)),    \
      BB_IM2COL_FUNC7)

#endif // _BB_IM2COL_H_
