#ifndef _BB_MXINT8QUANTIZE_H_
#define _BB_MXINT8QUANTIZE_H_
#include <bb_isa.h>

/* Bank rows of one packed MXINT8 row of length k: k codes, then k/32
 * exponent bytes, zero-padded to whole 16-byte bank rows. */
#define MXINT8_ROWS(k) ((k) / 16 + ((k) + 511) / 512)

/* Quantizes `rows` rows of k FP32 (row r at bank row in_base + r*k/4 of
 * `input`) to packed MXINT8 rows (row r at out_base + r*MXINT8_ROWS(k) of
 * `output`). */
static inline void bb_mxint8quant(uint32_t input, uint32_t output,
                                  uint64_t rows, uint32_t k, uint32_t in_base,
                                  uint32_t out_base) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) | BB_ITER(rows),
                            FIELD(k, 0, 15) | FIELD(in_base, 16, 29) |
                                FIELD(out_base, 30, 43),
                            FUNCT7(mxint8quantize, MXINT8QUANT));
}
#endif
