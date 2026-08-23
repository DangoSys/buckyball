#ifndef _BB_GEMMINI_LOOP_CONV_WS_H
#define _BB_GEMMINI_LOOP_CONV_WS_H

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_gemmini_loop_conv_ws_config_1(batch_size, in_dim, in_channels)      \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(batch_size, 0, 15) |                        \
                             FIELD(in_dim, 16, 31) |                           \
                             FIELD(in_channels, 32, 47)),                      \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_1))

#define bb_gemmini_loop_conv_ws_config_2(out_channels, out_dim, stride,        \
                                         padding)                              \
  BUCKYBALL_INSTRUCTION_R_R(0,                                                 \
                            (FIELD(out_channels, 0, 15) |                      \
                             FIELD(out_dim, 16, 31) | FIELD(stride, 32, 39) |  \
                             FIELD(padding, 40, 47)),                          \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_2))

#define bb_gemmini_loop_conv_ws_config_3(kernel_dim, pool_size, pool_stride,   \
                                         pool_padding)                         \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0,                                                                       \
      (FIELD(kernel_dim, 0, 7) | FIELD(pool_size, 8, 15) |                     \
       FIELD(pool_stride, 16, 23) | FIELD(pool_padding, 24, 31)),              \
      BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_3))

#define bb_gemmini_loop_conv_ws_config_4(addr_bias)                            \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_bias, 0, 38),                        \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_4))

#define bb_gemmini_loop_conv_ws_config_5(addr_input)                           \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_input, 0, 38),                       \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_5))

#define bb_gemmini_loop_conv_ws_config_6(addr_weight)                          \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_weight, 0, 38),                      \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_6))

#define bb_gemmini_loop_conv_ws_config_7(addr_output)                          \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(addr_output, 0, 38),                      \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_7))

#define bb_gemmini_loop_conv_ws_config_8(input_stride, weight_stride)          \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0, (FIELD(input_stride, 0, 31) | FIELD(weight_stride, 32, 63)),          \
      BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_8))

#define bb_gemmini_loop_conv_ws_config_9(output_stride)                        \
  BUCKYBALL_INSTRUCTION_R_R(0, FIELD(output_stride, 0, 31),                    \
                            BB_FUNC7(GEMMINI_LOOP_CONV_WS_CONFIG_9))

#define bb_gemmini_loop_conv_ws(bank_input, bank_weight, bank_output, no_bias) \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      0,                                                                       \
      (FIELD(bank_input, 0, 9) | FIELD(bank_weight, 10, 19) |                  \
       FIELD(bank_output, 20, 29) | FIELD(no_bias, 30, 30)),                   \
      BB_FUNC7(GEMMINI_LOOP_CONV_WS))

#endif
