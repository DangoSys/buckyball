#ifndef _BB_LAYERNORM_H_
#define _BB_LAYERNORM_H_
#include <bb_isa.h>

/* Bank rows of one FP32 row of length w: it starts on a bank row. */
#define LAYERNORM_ROWS(w) (((w) + 3) / 4)

/* The layer normalization of each of `rows` FP32 rows of length w
 * (16 <= w <= 2048), LAYERNORM_ROWS(w) bank rows each, from in_base of bank
 * `input` to out_base of bank `output` (another bank); the lanes after w of a
 * row's last bank row are ignored on input and zero on output. Bank `params`
 * holds from param_base gamma (LAYERNORM_ROWS(w) bank rows), then beta (as
 * many), then eps in lane 0 of one bank row. Every value finite. In FP32,
 * rounding to nearest, per row:
 *
 *   s    = (s_0 + s_1) + (s_2 + s_3), s_l the sum of x_i, i = l mod 4, in
 *          increasing i from 0
 *   mean = s * (1 / w)                 1 / w rounded once
 *   d_i  = x_i - mean
 *   q    = (q_0 + q_1) + (q_2 + q_3), q_l the sum of d_i * d_i likewise
 *   r    = 1 / sqrt(q * (1 / w) + eps)
 *   y_i  = (d_i * r) * gamma_i + beta_i
 *
 * With rms (RMSNorm), the mean is 0 (d_i = x_i) and there is no beta: bank
 * `params` holds gamma, then eps in lane 0 of the bank row after it, and
 * y_i = (x_i * r) * gamma_i. */
static inline void bb_layernorm(uint32_t input, uint32_t params,
                                uint32_t output, uint32_t rows, uint32_t w,
                                uint32_t in_base, uint32_t out_base,
                                uint32_t param_base, uint32_t rms) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(input) | BB_BANK1(params) | BB_BANK2(output) | BB_ITER(rows),
      FIELD(w, 0, 15) | FIELD(in_base, 16, 29) | FIELD(out_base, 30, 43) |
          FIELD(param_base, 44, 57) | FIELD(rms, 58, 58),
      FUNCT7(layernorm, LAYERNORM));
}
#endif
