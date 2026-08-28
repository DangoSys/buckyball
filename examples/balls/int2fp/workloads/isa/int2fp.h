#ifndef _BB_INT2FP_H_
#define _BB_INT2FP_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Dequantize INT32 accumulators with scales in absolute MMIO byte addresses.
// rs2[12:0] = activation Da address, rs2[25:13] = weight Dw address.
#define BB_INT2FP_RS1(bank_id, wr_bank_id, iter)                               \
  (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter))
#define BB_INT2FP_RS2(act_scale_addr, weight_scale_addr)                       \
  (FIELD((uint64_t)(act_scale_addr), 0, 12) |                                  \
   FIELD((uint64_t)(weight_scale_addr), 13, 25))

#define bb_int2fp_tensor(bank_id, wr_bank_id, iter, act_scale_addr,            \
                         weight_scale_addr)                                    \
  BUCKYBALL_INSTRUCTION_R_R(BB_INT2FP_RS1(bank_id, wr_bank_id, iter),          \
                            BB_INT2FP_RS2(act_scale_addr, weight_scale_addr),  \
                            BB_FUNC7(INT2FP_TENSOR))

#define bb_int2fp_channel(bank_id, wr_bank_id, iter, act_scale_addr,           \
                          weight_scale_addr)                                   \
  BUCKYBALL_INSTRUCTION_R_R(BB_INT2FP_RS1(bank_id, wr_bank_id, iter),          \
                            BB_INT2FP_RS2(act_scale_addr, weight_scale_addr),  \
                            BB_FUNC7(INT2FP_CHANNEL))

#endif // _BB_INT2FP_H_
