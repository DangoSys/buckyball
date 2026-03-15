#ifndef _BB_GEMMINI_LOOP_WS_H
#define _BB_GEMMINI_LOOP_WS_H

#include "isa.h"

#define BB_GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNC7 80
#define BB_GEMMINI_LOOP_WS_CONFIG_ADDR_A_FUNC7 81
#define BB_GEMMINI_LOOP_WS_CONFIG_ADDR_B_FUNC7 82
#define BB_GEMMINI_LOOP_WS_CONFIG_ADDR_D_FUNC7 83
#define BB_GEMMINI_LOOP_WS_CONFIG_ADDR_C_FUNC7 84
#define BB_GEMMINI_LOOP_WS_CONFIG_STRIDES_AB_FUNC7 85
#define BB_GEMMINI_LOOP_WS_CONFIG_STRIDES_DC_FUNC7 86
#define BB_GEMMINI_LOOP_WS_FUNC7 87

#define bb_gemmini_loop_ws_config_bounds(max_i, max_j, max_k)                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(max_k, 0, 15) | FIELD(max_j, 16, 31) | FIELD(max_i, 32, 47)),  \
      BB_GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNC7)

#define bb_gemmini_loop_ws_config_addr_a(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_GEMMINI_LOOP_WS_CONFIG_ADDR_A_FUNC7)

#define bb_gemmini_loop_ws_config_addr_b(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_GEMMINI_LOOP_WS_CONFIG_ADDR_B_FUNC7)

#define bb_gemmini_loop_ws_config_addr_d(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_GEMMINI_LOOP_WS_CONFIG_ADDR_D_FUNC7)

#define bb_gemmini_loop_ws_config_addr_c(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_GEMMINI_LOOP_WS_CONFIG_ADDR_C_FUNC7)

#define bb_gemmini_loop_ws_config_strides_ab(stride_a, stride_b)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(stride_a, 0, 31) | FIELD(stride_b, 32, 63)),                   \
      BB_GEMMINI_LOOP_WS_CONFIG_STRIDES_AB_FUNC7)

#define bb_gemmini_loop_ws_config_strides_dc(stride_d, stride_c)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(stride_d, 0, 31) | FIELD(stride_c, 32, 63)),                   \
      BB_GEMMINI_LOOP_WS_CONFIG_STRIDES_DC_FUNC7)

#define bb_gemmini_loop_ws(bank_a, bank_b, bank_c, low_d)                      \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(bank_a, 0, 9) | FIELD(bank_b, 10, 19) |     \
                             FIELD(bank_c, 20, 29) | FIELD(low_d, 30, 30)),    \
                            BB_GEMMINI_LOOP_WS_FUNC7)

#endif
