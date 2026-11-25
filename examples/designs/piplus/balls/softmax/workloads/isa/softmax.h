#ifndef _BB_SOFTMAX_H_
#define _BB_SOFTMAX_H_
#include <bb_isa.h>

/* Bank rows of one FP32 row of length w: it starts on a bank row. */
#define SOFTMAX_ROWS(w) (((w) + 3) / 4)

/* The softmax of each of `rows` FP32 rows of length w (16 <= w <= 2048),
 * SOFTMAX_ROWS(w) bank rows each, from in_base of bank `input` to out_base of
 * bank `output` (another bank); the lanes after w of a row's last bank row
 * are ignored on input and zero on output. The row is first scaled by
 * 2^scale_log2 (-126 <= scale_log2 <= 127), x_i = in_i * 2^scale_log2; every
 * x_i is finite or -inf, a row holding a finite one. In FP32, rounding to
 * nearest, per row:
 *
 *   x_i = in_i * 2^scale_log2
 *   m   = the maximum of the row
 *   u_i = (x_i - m) * log2(e)
 *   e_i = 0 if u_i < -126, else P(f) * 2^n with n = floor(u_i), f = u_i - n
 *         and P(f) = ((((((c6 f + c5) f + c4) f + c3) f + c2) f + c1) f + 1)
 *   s   = (s_0 + s_1) + (s_2 + s_3), s_l the sum of e_i, i = l mod 4, in
 *         increasing i from 0
 *   y_i = e_i * (1 / s)
 *
 * with the constants of emu/src/softmax.rs. */
static inline void bb_softmax(uint32_t input, uint32_t output, uint32_t rows,
                              uint32_t w, uint32_t in_base, uint32_t out_base,
                              int32_t scale_log2) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input) | BB_BANK2(output) | BB_ITER(rows),
                            FIELD(w, 0, 15) | FIELD(in_base, 16, 29) |
                                FIELD(out_base, 30, 43) |
                                FIELD((uint32_t)scale_log2 & 0xff, 44, 51),
                            FUNCT7(softmax, SOFTMAX));
}
#endif
