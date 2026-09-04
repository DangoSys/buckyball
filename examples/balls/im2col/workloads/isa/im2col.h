#ifndef _BB_IM2COL_H_
#define _BB_IM2COL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_im2col(input_bank, output_bank, input_size, kernel, stride,         \
                  padding, input_base, lane, start_row, start_col,             \
                  window_start, window_count)                                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK2(output_bank) | BB_ITER(input_size),      \
      FIELD(kernel, 0, 7) | FIELD(stride, 8, 15) | FIELD(padding, 16, 23) |    \
          FIELD(start_col, 24, 31) | FIELD(start_row, 32, 39) |                \
          FIELD(input_base, 40, 45) | FIELD(lane, 46, 49) |                    \
          FIELD(window_start, 50, 55) | FIELD(window_count, 56, 62),           \
      BB_FUNC7(IM2COL))

#endif // _BB_IM2COL_H_
