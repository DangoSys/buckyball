#ifndef _BB_MVOUT_H_
#define _BB_MVOUT_H_

#include "isa.h"

#define BB_MVOUT_FUNC7 25

#define bb_mvout(mem_addr, bank_id, depth, stride)                             \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_RD0 | BB_ITER(depth)),     \
                            (FIELD(mem_addr, 0, 38) | FIELD(stride, 39, 57)),  \
                            BB_MVOUT_FUNC7)

#endif // _BB_MVOUT_H_
