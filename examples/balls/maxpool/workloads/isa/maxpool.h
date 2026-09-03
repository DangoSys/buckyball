#ifndef _BB_MAXPOOL_H_
#define _BB_MAXPOOL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_maxpool(input_bank, output_bank, input_side, output_side, kernel,   \
                   stride, padding, input_base, output_base, output_stride)    \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK2(output_bank) |                           \
          BB_ITER((output_side) * (output_side)),                              \
      FIELD(input_side, 0, 7) | FIELD(output_side, 8, 15) |                    \
          FIELD(kernel, 16, 23) | FIELD(stride, 24, 31) |                      \
          FIELD(padding, 32, 39) | FIELD(input_base, 40, 47) |                 \
          FIELD(output_base, 48, 55) | FIELD(output_stride, 56, 63),           \
      BB_FUNC7(MAXPOOL))

#endif // _BB_MAXPOOL_H_
