#ifndef _BB_MAXPOOL_H_
#define _BB_MAXPOOL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_maxpool(input_bank, output_bank, input_side, output_side, kernel,   \
                   stride, padding, input_base, output_base, output_stride,    \
                   start_row, start_col)                                       \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK2(output_bank) |                           \
          BB_ITER((output_side) * (output_side)),                              \
      FIELD(input_side, 0, 3) | FIELD(output_side, 4, 7) |                     \
          FIELD(kernel, 8, 11) | FIELD(stride, 12, 15) |                       \
          FIELD(padding, 16, 19) | FIELD(input_base, 20, 25) |                 \
          FIELD(output_base, 26, 31) | FIELD(output_stride, 32, 37) |          \
          FIELD(start_row, 38, 41) | FIELD(start_col, 42, 45),                 \
      BB_FUNC7(MAXPOOL))

#endif // _BB_MAXPOOL_H_
