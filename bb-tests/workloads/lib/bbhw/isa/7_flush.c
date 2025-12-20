#ifndef _BB_FLUSH_H_
#define _BB_FLUSH_H_

#include "isa.h"

#define BB_FLUSH_FUNC7 7

#define bb_flush() BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_FLUSH_FUNC7)

#endif // _BB_FLUSH_H_
