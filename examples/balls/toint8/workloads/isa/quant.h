#ifndef _BB_QUANT_H_
#define _BB_QUANT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

static inline void bb_quant_f32_to_i8(uint64_t input_bank, uint64_t output_bank,
                                      uint64_t iter, float scale) {
  union {
    float f;
    uint32_t u;
  } bits = {.f = scale};
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK2(output_bank) |
                                BB_ITER(iter),
                            bits.u, BB_FUNC7(QUANT_F32_TO_I8));
}

static inline void bb_quant_i32_to_i8(uint64_t input_bank, uint64_t scale_bank,
                                      uint64_t output_bank, uint64_t iter,
                                      uint64_t output_base,
                                      uint64_t output_width,
                                      uint64_t output_height,
                                      uint64_t output_stride, int relu) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK1(scale_bank) |
                                BB_BANK2(output_bank) | BB_ITER(iter),
                            (output_base << 1) | (output_width << 8) |
                                (output_height << 15) | (output_stride << 22) |
                                (uint64_t)(relu != 0),
                            BB_FUNC7(QUANT_I32_TO_I8));
}

#endif // _BB_QUANT_H_
