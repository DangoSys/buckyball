#ifndef _BB_FENCE_H_
#define _BB_FENCE_H_

#include "isa.h"

#define BB_FENCE_FUNC7 31

#define bb_fence() BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_FENCE_FUNC7)

#endif // _BB_FENCE_H_
