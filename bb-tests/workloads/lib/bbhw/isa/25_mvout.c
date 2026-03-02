#ifndef _BB_MVOUT_H_
#define _BB_MVOUT_H_

#include "isa.h"

#define BB_MVOUT_FUNC7 25

#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_RD0 | FIELD(mem_addr, 27, 58)),                  \
      (FIELD(depth, 0, 9) | FIELD(stride, 10, 28)), BB_MVOUT_FUNC7)

#endif // _BB_MVOUT_H_
