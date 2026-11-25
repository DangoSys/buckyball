#ifndef BB_INTNORM_H
#define BB_INTNORM_H
#include <bb_isa.h>

// Integer LayerNorm: FP32 bit patterns -> Q16 arithmetic -> packed MXINT8.
// K is a multiple of 512. Coefficients start with EPS in Q32 (u64) and
// two zero words, followed by K Q16 gamma and K Q16 beta words.
// Output rows contain K codes then K/32 exponents. The output bank also
// reserves K*4 scratch bytes immediately after all packed output rows.
static inline void bb_intnorm(uint32_t input, uint32_t coefficients,
                              uint32_t output, uint32_t rows, uint32_t k,
                              uint32_t in_base, uint32_t coefficient_base,
                              uint32_t out_base) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK1(coefficients) |
                                BB_BANK2(output) | BB_ITER(rows),
                            FIELD(k, 0, 15) | FIELD(in_base, 16, 29) |
                                FIELD(out_base, 30, 43) |
                                FIELD(coefficient_base, 44, 57),
                            FUNCT7(intnorm, INTNORM));
}
#endif
