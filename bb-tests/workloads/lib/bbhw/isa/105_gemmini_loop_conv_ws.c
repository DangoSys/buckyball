#ifndef _BB_GEMMINI_LOOP_CONV_WS_H
#define _BB_GEMMINI_LOOP_CONV_WS_H

#include "isa.h"

#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_1_FUNC7 96
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_2_FUNC7 97
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_3_FUNC7 98
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_4_FUNC7 99
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_5_FUNC7 100
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_6_FUNC7 101
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_7_FUNC7 102
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_8_FUNC7 103
#define BB_GEMMINI_LOOP_CONV_WS_CONFIG_9_FUNC7 104
#define BB_GEMMINI_LOOP_CONV_WS_FUNC7 105

#define bb_gemmini_loop_conv_ws_config_1(batch_size, in_dim, in_channels)      \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(batch_size, 0, 15) |                        \
                             FIELD(in_dim, 16, 31) |                           \
                             FIELD(in_channels, 32, 47)),                      \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_1_FUNC7)

#define bb_gemmini_loop_conv_ws_config_2(out_channels, out_dim, stride,        \
                                         padding)                              \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(out_channels, 0, 15) |                      \
                             FIELD(out_dim, 16, 31) | FIELD(stride, 32, 39) |  \
                             FIELD(padding, 40, 47)),                          \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_2_FUNC7)

#define bb_gemmini_loop_conv_ws_config_3(kernel_dim, pool_size, pool_stride,   \
                                         pool_padding)                         \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0,                                                                       \
      (FIELD(kernel_dim, 0, 7) | FIELD(pool_size, 8, 15) |                     \
       FIELD(pool_stride, 16, 23) | FIELD(pool_padding, 24, 31)),              \
      BB_GEMMINI_LOOP_CONV_WS_CONFIG_3_FUNC7)

#define bb_gemmini_loop_conv_ws_config_4(addr_bias)                            \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_bias, 0, 38),                        \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_4_FUNC7)

#define bb_gemmini_loop_conv_ws_config_5(addr_input)                           \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_input, 0, 38),                       \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_5_FUNC7)

#define bb_gemmini_loop_conv_ws_config_6(addr_weight)                          \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_weight, 0, 38),                      \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_6_FUNC7)

#define bb_gemmini_loop_conv_ws_config_7(addr_output)                          \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_output, 0, 38),                      \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_7_FUNC7)

#define bb_gemmini_loop_conv_ws_config_8(input_stride, weight_stride)          \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(input_stride, 0, 31) | FIELD(weight_stride, 32, 63)),          \
      BB_GEMMINI_LOOP_CONV_WS_CONFIG_8_FUNC7)

#define bb_gemmini_loop_conv_ws_config_9(output_stride)                        \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(output_stride, 0, 31),                    \
                            BB_GEMMINI_LOOP_CONV_WS_CONFIG_9_FUNC7)

#define bb_gemmini_loop_conv_ws(bank_input, bank_weight, bank_output, no_bias) \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0,                                                                       \
      (FIELD(bank_input, 0, 9) | FIELD(bank_weight, 10, 19) |                  \
       FIELD(bank_output, 20, 29) | FIELD(no_bias, 30, 30)),                   \
      BB_GEMMINI_LOOP_CONV_WS_FUNC7)

#endif
