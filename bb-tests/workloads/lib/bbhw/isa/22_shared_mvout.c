#ifndef _BB_SHARED_MVOUT_H_
#define _BB_SHARED_MVOUT_H_

#include "isa.h"

#define BB_SHARED_MVOUT_FUNC7 22

#define bb_shared_mvout(mem_addr, bank_id, depth, stride)                             \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_RD0 | FIELD(mem_addr, 27, 58)),                  \
      (FIELD(depth, 0, 9) | FIELD(stride, 10, 28)), BB_SHARED_MVOUT_FUNC7)

#endif // _BB_SHARED_MVOUT_H_
