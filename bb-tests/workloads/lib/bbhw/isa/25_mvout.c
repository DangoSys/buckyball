#ifndef _BB_MVOUT_H_
#define _BB_MVOUT_H_

#include "isa.h"

#define BB_MVOUT_FUNC7 25

#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      FIELD(mem_addr, 0, 31),                                                  \
      (FIELD(bank_id, 0, 7) | FIELD(depth, 8, 23) | FIELD(stride, 24, 33)),    \
      BB_MVOUT_FUNC7)

#endif // _BB_MVOUT_H_
