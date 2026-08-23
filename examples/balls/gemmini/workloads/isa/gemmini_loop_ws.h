#ifndef _BB_GEMMINI_LOOP_WS_H
#define _BB_GEMMINI_LOOP_WS_H

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_gemmini_loop_ws_config_bounds(max_i, max_j, max_k)                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(max_k, 0, 15) | FIELD(max_j, 16, 31) | FIELD(max_i, 32, 47)),  \
      BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_BOUNDS))

#define bb_gemmini_loop_ws_config_addr_a(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_ADDR_A))

#define bb_gemmini_loop_ws_config_addr_b(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_ADDR_B))

#define bb_gemmini_loop_ws_config_addr_d(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_ADDR_D))

#define bb_gemmini_loop_ws_config_addr_c(addr)                                 \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr, 0, 38),                             \
                            BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_ADDR_C))

#define bb_gemmini_loop_ws_config_strides_ab(stride_a, stride_b)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(stride_a, 0, 31) | FIELD(stride_b, 32, 63)),                   \
      BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_STRIDES_AB))

#define bb_gemmini_loop_ws_config_strides_dc(stride_d, stride_c)               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(stride_d, 0, 31) | FIELD(stride_c, 32, 63)),                   \
      BB_FUNC7(GEMMINI_LOOP_WS_CONFIG_STRIDES_DC))

#define bb_gemmini_loop_ws(bank_a, bank_b, bank_c, low_d)                      \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(bank_a, 0, 9) | FIELD(bank_b, 10, 19) |     \
                             FIELD(bank_c, 20, 29) | FIELD(low_d, 30, 30)),    \
                            BB_FUNC7(GEMMINI_LOOP_WS))

#endif
