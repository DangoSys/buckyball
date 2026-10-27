#ifndef _BB_INT2FP_H_
#define _BB_INT2FP_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Group g of every operand bank is channel block g (channels 16g..16g+15).
// Input, scale, and output banks have the same number of groups; every group
// uses the same rows. In group g, input row r holds 4 INT32 lanes, scale row
// r % 4 holds their scales, and output row r holds the 4 FP32 results.

#define bb_int32_to_fp32(input_bank, scale_bank, output_bank, iter, relu)      \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK1(scale_bank) |      \
                                BB_BANK2(output_bank) | BB_ITER(iter),         \
                            FIELD((uint64_t)(relu), 0, 0),                     \
                            FUNCT7(int2fp, INT32_TO_FP32))

#endif // _BB_INT2FP_H_
