#ifndef _BB_MXINT8MM_H_
#define _BB_MXINT8MM_H_
#include <bb_isa.h>

/* Bank rows of one packed MXINT8 row of length k: k codes, then k/32
 * exponent bytes, zero-padded to whole 16-byte bank rows. */
#define MXINT8_ROWS(k) ((k) / 16 + ((k) + 511) / 512)

/* C[m,n] (FP32, n/4 bank rows per row, row r at c_base + r * c_pitch, c_pitch
 * >= n/4) = A[m,k] . B[n,k]^T with A (from a_base) and B (from bank row 0)
 * packed MXINT8 rows. Each element starts
 * at +0.0, or with accumulate at the old C of the destination; with bias,
 * bias[col] is then added to it (FP32), bias the n values of the n/4 bank rows
 * before C's window (from c_base - n/4, c_base >= n/4) in the old destination;
 * then the block terms, in block order. The instruction defines a new value
 * of bank c: the bank rows it does not write are undefined afterwards. With
 * update it writes C in place and the other bank rows of c stay as they
 * were. */
static inline void bb_mxint8mm(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                               uint32_t n, uint64_t k, uint32_t a_base,
                               uint32_t c_pitch, uint32_t c_base,
                               uint32_t accumulate, uint32_t bias,
                               uint32_t update) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) | BB_ITER(k) |
          (accumulate || bias ? BB_READ_WR_OLD : 0) |
          (update ? BB_UPDATE_WR : 0),
      FIELD(m, 0, 9) | FIELD(n, 10, 19) | FIELD(a_base, 20, 33) |
          FIELD(c_pitch, 34, 47) | FIELD(c_base, 48, 61) |
          FIELD(accumulate, 62, 62) | FIELD(bias, 63, 63),
      FUNCT7(mxint8mm, MXINT8MM));
}
#endif
