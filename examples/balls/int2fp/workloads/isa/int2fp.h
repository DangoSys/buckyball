#ifndef _BB_INT2FP_H_
#define _BB_INT2FP_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define bb_int32_to_fp32(input_bank, scale_bank, output_bank, iter, relu)      \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK1(scale_bank) |      \
                                BB_BANK2(output_bank) | BB_ITER(iter),         \
                            FIELD((uint64_t)(relu), 0, 0),                     \
                            BB_FUNC7(INT32_TO_FP32))

#endif // _BB_INT2FP_H_
