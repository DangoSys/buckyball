#ifndef _BB_BBUS_CONFIG_H_
#define _BB_BBUS_CONFIG_H_

#include "isa.h"

#define BB_BBUS_CONFIG_FUNC7 39

#define bb_bbus_config(src_bid, dst_bid, enable)                               \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(src_bid,0, 5) | FIELD(dst_bid, 6, 11) | \
                             FIELD(enable, 12, 12)),                           \
                            BB_BBUS_CONFIG_FUNC7)

#endif // _BB_BBUS_CONFIG_H_
