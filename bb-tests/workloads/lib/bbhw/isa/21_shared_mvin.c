#ifndef _BB_SHARED_MVIN_H_
#define _BB_SHARED_MVIN_H_

#include "isa.h"

#define BB_SHARED_MVIN_FUNC7 21

#define bb_shared_mvin(mem_addr, bank_id, depth, stride)                       \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_WR | BB_ITER(depth)),      \
                            (FIELD(mem_addr, 0, 38) | FIELD(stride, 39, 57)),  \
                            BB_SHARED_MVIN_FUNC7)

#endif // _BB_SHARED_MVIN_H_
