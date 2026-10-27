#ifndef _BB_QUANT_H_
#define _BB_QUANT_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Group g of every operand bank is channel block g (channels 16g..16g+15).
// The input, scale (I32 only), and output banks have the same number of
// groups; every group uses the same rows, so one instruction converts all
// channel blocks. In group g, input rows hold 4 FP32/INT32 lanes (four rows
// per pixel), scale rows 0..3 hold the 16 per-channel scales of block g, and
// each output row holds the 16 INT8 channels of one pixel.

static inline void bb_quant_f32_to_i8(uint64_t input_bank, uint64_t output_bank,
                                      uint64_t iter, float scale) {
  union {
    float f;
    uint32_t u;
  } bits = {.f = scale};
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK2(output_bank) |
                                BB_ITER(iter),
                            bits.u, FUNCT7(toint8, QUANT_F32_TO_I8));
}

static inline void bb_quant_i32_to_i8(uint64_t input_bank, uint64_t scale_bank,
                                      uint64_t output_bank, uint64_t iter,
                                      uint64_t input_base, uint64_t output_base,
                                      uint64_t output_width,
                                      uint64_t output_height,
                                      uint64_t output_stride, int relu) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK1(scale_bank) |
                                BB_BANK2(output_bank) | BB_ITER(iter),
                            (output_base << 1) | (output_width << 8) |
                                (output_height << 15) | (output_stride << 22) |
                                (input_base << 29) | (uint64_t)(relu != 0),
                            FUNCT7(toint8, QUANT_I32_TO_I8));
}

#endif // _BB_QUANT_H_
