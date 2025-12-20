#ifndef _BB_CONV_H_
#define _BB_CONV_H_

#include "isa.h"

#define BB_CONV_FUNC7 43

#define bb_conv(ifmap_addr, weight_addr, ofmap_addr, iter, in_height,          \
                in_width, kernel_h, kernel_w)                                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (FIELD(ifmap_addr, 0, 14) | FIELD(weight_addr, 15, 29)),                 \
      (FIELD(ofmap_addr, 0, 14) | FIELD(iter, 15, 24) |                        \
       FIELD(in_height, 25, 40) | FIELD(in_width, 41, 56) |                    \
       FIELD(kernel_h, 57, 64)),                                               \
      BB_CONV_FUNC7)

#endif // _BB_CONV_H_
