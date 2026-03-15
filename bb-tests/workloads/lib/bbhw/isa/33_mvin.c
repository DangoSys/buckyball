#ifndef _BB_MVIN_H_
#define _BB_MVIN_H_

#include "isa.h"

#define BB_MVIN_FUNC7 33

#define bb_mvin(mem_addr, bank_id, depth, stride)                              \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_ITER(depth)),              \
                            (FIELD(mem_addr, 0, 38) | FIELD(stride, 39, 57)),  \
                            BB_MVIN_FUNC7)

#endif // _BB_MVIN_H_
