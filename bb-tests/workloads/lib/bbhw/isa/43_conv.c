#ifndef _BB_CONV_H_
#define _BB_CONV_H_

#include "isa.h"

#define BB_CONV_FUNC7 43

#define bb_conv(ifmap_bank_id, weight_bank_id, ofmap_bank_id, iter, in_height,          \
                in_width, kernel_h, kernel_w)                                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(ifmap_bank_id, 0, 7) | FIELD(weight_bank_id, 8, 15)),                 \
      (FIELD(ofmap_bank_id, 0, 7) | FIELD(iter, 8, 23) |                        \
       FIELD(in_height, 24, 39) | FIELD(in_width, 40, 55) |                    \
       FIELD(kernel_h, 56, 63)),                                               \
      BB_CONV_FUNC7)

#endif // _BB_CONV_H_
