#ifndef _BB_MXINT8MM_H_
#define _BB_MXINT8MM_H_
#include <bb_isa.h>

/* Bank rows of one packed MXINT8 row of length k: k codes, then k/32
 * exponent bytes, zero-padded to whole 16-byte bank rows. */
#define MXINT8_ROWS(k) ((k) / 16 + ((k) + 511) / 512)

/* C[m,n] (FP32, n/4 bank rows per row from c_base) = A[m,k] . B[n,k]^T with
 * A (from a_base) and B (from b_base) packed MXINT8 rows. With accumulate,
 * the products add onto the old C of the destination. */
static inline void bb_mxint8mm(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                               uint32_t n, uint64_t k, uint32_t a_base,
                               uint32_t b_base, uint32_t c_base,
                               uint32_t accumulate) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) |
                                BB_ITER(k) | (accumulate ? BB_READ_WR_OLD : 0),
                            FIELD(m, 0, 9) | FIELD(n, 10, 19) |
                                FIELD(a_base, 20, 33) | FIELD(b_base, 34, 47) |
                                FIELD(c_base, 48, 61) |
                                FIELD(accumulate, 62, 62),
                            FUNCT7(mxint8mm, MXINT8MM));
}
#endif
