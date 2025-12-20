#ifndef _BB_BBUS_CONFIG_H_
#define _BB_BBUS_CONFIG_H_

#include "isa.h"

#define BB_BBUS_CONFIG_FUNC7 39

#define bb_bbus_config(src_bid, dst_bid, enable)                               \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(src_bid, 25, 30) | FIELD(dst_bid, 31, 36) | \
                             FIELD(enable, 37, 37)),                           \
                            BB_BBUS_CONFIG_FUNC7)

#endif // _BB_BBUS_CONFIG_H_
