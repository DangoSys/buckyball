#ifndef _BB_MXINT8QUANT_H_
#define _BB_MXINT8QUANT_H_
#include <bb_isa.h>

/* Bank rows of one packed MXINT8 row of length k: k codes, then k/32
 * exponent bytes, zero-padded to whole 16-byte bank rows. */
#define MXINT8_ROWS(k) ((k) / 16 + ((k) + 511) / 512)

/* Quantizes `rows` FP32 rows of length k (k/4 bank rows each, from in_base)
 * to packed MXINT8 rows (MXINT8_ROWS(k) bank rows each, from out_base). */
static inline void bb_mxint8quant(uint32_t input, uint32_t output,
                                  uint32_t rows, uint32_t k, uint32_t in_base,
                                  uint32_t out_base) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) | BB_ITER(rows),
                            FIELD(k, 0, 15) | FIELD(in_base, 16, 29) |
                                FIELD(out_base, 30, 43),
                            FUNCT7(mxint8quant, MXINT8QUANT));
}
#endif
