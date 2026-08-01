#ifndef _BB_INT2FP_H_
#define _BB_INT2FP_H_

#include "isa.h"

#define BB_INT2FP_FUNC7 52

#define BB_INT_OUTPUT_FP32 0
#define BB_INT_OUTPUT_INT8 1
#define BB_INT_OUTPUT_MODE(mode) FIELD((uint64_t)(mode), 32, 33)

// Convert integer data using an explicit output format.
// Encoding: rs1 = banks | iter
//           rs2[31:0]  = scale_fp32
//           rs2[33:32] = output_mode
#define bb_int_convert(bank_id, wr_bank_id, iter, output_mode, scale_fp32)     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),              \
      (FIELD((uint64_t)(scale_fp32), 0, 31) |                                  \
       BB_INT_OUTPUT_MODE(output_mode)),                                       \
      BB_INT2FP_FUNC7)

// bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)
// scale_fp32 is a 32-bit FP32 value passed as uint32_t bit pattern
// Encoding: rs1 = banks | iter
//           rs2 = FIELD(scale_fp32, 0, 31)
#define bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)                       \
  bb_int_convert(bank_id, wr_bank_id, iter, BB_INT_OUTPUT_FP32, scale_fp32)

// Requantize four INT32 bank groups into one packed INT8 bank group.
#define bb_int32_to_int8(bank_id, wr_bank_id, iter, scale_fp32)                \
  bb_int_convert(bank_id, wr_bank_id, iter, BB_INT_OUTPUT_INT8, scale_fp32)

#endif // _BB_INT2FP_H_
