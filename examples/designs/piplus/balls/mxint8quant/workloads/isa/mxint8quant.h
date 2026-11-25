#ifndef _BB_MXINT8QUANT_H_
#define _BB_MXINT8QUANT_H_
#include <bb_isa.h>

/* Bank rows of one packed MXINT8 row of length k: k codes, then k/32
 * exponent bytes, zero-padded to whole 16-byte bank rows. */
#define MXINT8_ROWS(k) ((k) / 16 + ((k) + 511) / 512)

/* Quantizes `rows` FP32 rows of length k to packed MXINT8 rows
 * (MXINT8_ROWS(k) bank rows each, row after row from out_base). Row i is the
 * k/4 bank rows from in_base + i * in_pitch (in_pitch >= k/4). With columns
 * (rows a multiple of 4), the input is a row-major FP32 matrix [k][rows],
 * its row r the rows/4 bank rows from in_base + r * in_pitch (in_pitch >=
 * rows/4), and output row c is its column c, quantized along k: the
 * transpose of a matrix in a bank, or of its columns from 4 * g with in_base
 * + g. The instruction defines a new value of the output bank: the bank rows
 * it does not write are undefined afterwards. With update it writes in place
 * and the other bank rows of the output stay as they were. */
static inline void bb_mxint8quant(uint32_t input, uint32_t output,
                                  uint32_t rows, uint32_t k, uint32_t in_base,
                                  uint32_t out_base, uint32_t columns,
                                  uint64_t in_pitch, uint32_t update) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(input) | BB_BANK2(output) | BB_ITER(rows) |
          (update ? BB_UPDATE_WR : 0),
      FIELD(k, 0, 15) | FIELD(in_base, 16, 29) | FIELD(out_base, 30, 43) |
          FIELD(columns, 44, 44) | FIELD(in_pitch, 45, 58),
      FUNCT7(mxint8quant, MXINT8QUANT));
}
#endif
