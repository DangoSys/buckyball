#ifndef _BB_MVIN_H_
#define _BB_MVIN_H_

#include "isa.h"

#define BB_MVIN_FUNC7 24

#define bb_mvin(mem_addr, bank_id, depth, stride)                              \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      FIELD(mem_addr, 0, 31),                                                  \
      (FIELD(bank_id, 0, 7) | FIELD(depth, 8, 23) | FIELD(stride, 24, 33)),    \
      BB_MVIN_FUNC7)

#endif // _BB_MVIN_H_
